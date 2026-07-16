/**
 * Pico PAL CVBS 视频协处理器
 *
 * PIO+DMA 连续 PAL 输出 @ 8 MHz: 40000B/frame = 40ms = 25fps
 * SPI 从机接收 RDK X5 帧数据, 双缓冲无撕裂
 *
 * 烧写: 按住 BOOTSEL → USB → 拖 .uf2 → RPI-RP2
 */

#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/spi.h"
#include "hardware/dma.h"
#include "hardware/clocks.h"
#include "hardware/watchdog.h"
#include "pal_video.pio.h"
#include <string.h>

// ─── 看门狗超时 (ms) ─────────────────────────────────────────────────
#define WDT_TIMEOUT_MS  2000   // 2s 无帧 → 自动复位

// ─── 引脚 ──────────────────────────────────────────────────────────
#define PIN_VIDEO   0
#define SPI_PORT    spi0
#define PIN_SCK     6
#define PIN_MOSI    4
#define PIN_CS      5

// ─── PAL 参数 (8 MHz 编码) ──────────────────────────────────────────
#define LINE_BYTES    64
#define LINES_FRAME   625
#define FRAME_BYTES   (LINE_BYTES * LINES_FRAME)  // 40000

// ─── 双缓冲 ─────────────────────────────────────────────────────────
static uint8_t __attribute__((aligned(4))) buf_a[FRAME_BYTES];
static uint8_t __attribute__((aligned(4))) buf_b[FRAME_BYTES];

static uint8_t * volatile active_buf = buf_a;   // 指针本身 volatile — ISR 可见
static int dma_chan = -1;

// ─── DMA IRQ: 每帧结束重启, 从 active_buf 继续输出 ─────────────────
static void dma_handler(void)
{
    if (dma_channel_get_irq0_status(dma_chan)) {
        dma_channel_acknowledge_irq0(dma_chan);
        dma_channel_set_read_addr(dma_chan, active_buf, false);
        dma_channel_set_trans_count(dma_chan, FRAME_BYTES, true);
    }
}

// ─── 启动 PIO + DMA 视频输出 ────────────────────────────────────────
static void video_start(void)
{
    PIO pio = pio0;
    uint sm = 0;

    uint offset = pio_add_program(pio, &pal_video_program);
    pal_video_program_init(pio, sm, offset, PIN_VIDEO, 14.0f);

    // DMA: memory → PIO TX FIFO, DREQ 驱动
    dma_chan = dma_claim_unused_channel(true);
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);

    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_8);
    channel_config_set_read_increment(&cfg, true);   // 递增读内存
    channel_config_set_write_increment(&cfg, false);  // 固定写 FIFO
    channel_config_set_dreq(&cfg, pio_get_dreq(pio, sm, true));

    dma_channel_configure(
        dma_chan, &cfg,
        &pio->txf[sm],      // 写 PIO TX FIFO
        active_buf,          // 读帧缓冲
        FRAME_BYTES,
        false
    );

    dma_channel_set_irq0_enabled(dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_handler);
    irq_set_enabled(DMA_IRQ_0, true);

    pio_sm_set_enabled(pio, sm, true);
    dma_channel_start(dma_chan);
}

// ─── SPI 从机: SCLK 间隔检测收帧 (无需 CS) ─────────────────────────
// X5 的硬件 CS 不工作, 改用 SCLK 空闲间隔 (>5ms) 检测帧边界
// Python 端每 40ms 发一帧 (20ms SPI 传输 + 20ms SCLK 空闲)
static void recv_frame(uint8_t *dst)
{
    // 等当前传输结束 (SCLK 回到空闲低电平)
    while (gpio_get(PIN_SCK));

    // 等 SCLK 连续空闲 > 5ms, 确认为帧间隔
    uint32_t idle_start = time_us_32();
    while (time_us_32() - idle_start < 5000) {
        if (gpio_get(PIN_SCK))  // SCLK 又活跃了, 还没到帧间隔
            idle_start = time_us_32();
    }

    // 等下一帧开始 (SCLK 起跳)
    while (!gpio_get(PIN_SCK));

    // 收完整帧
    spi_read_blocking(SPI_PORT, 0, dst, FRAME_BYTES);
}

// ─── 主函数 ────────────────────────────────────────────────────────
int main(void)
{
    // 126 MHz / 14 = 9.000 MHz PIO → 8/9 overhead = 8.000 MHz 精确视频时钟
    set_sys_clock_khz(126000, true);
    stdio_init_all();

    // 初始化为空帧 (纯黑 + 正常同步: 由 RDK X5 发送首帧覆盖)
    memset(buf_a, 0x00, FRAME_BYTES);
    memset(buf_b, 0x00, FRAME_BYTES);

    // SPI 从机: CPOL=0 CPHA=0 MSB-first 8bit
    spi_init(SPI_PORT, 16 * 1000 * 1000);
    spi_set_format(SPI_PORT, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);
    gpio_set_function(PIN_MOSI, GPIO_FUNC_SPI);
    gpio_set_function(PIN_SCK,  GPIO_FUNC_SPI);
    gpio_set_function(PIN_CS,   GPIO_FUNC_SPI);
    gpio_pull_down(PIN_CS);  // 内部下拉 → CS=0V (无需外部信号)
    // ★ 拔掉 X5 Pin24 → Pico GP5 的杜邦线, 否则 3.3V 会覆盖下拉
    spi_set_slave(SPI_PORT, true);

    // 启动 PIO + DMA 视频 (初始输出空白帧, 直到收到首帧)
    video_start();

    // 使能看门狗: X5 停止发帧 > 2s → Pico 自动复位
    watchdog_enable(WDT_TIMEOUT_MS, true);

    // 主循环: 收帧 → 交换缓冲 → 喂狗
    while (1) {
        uint8_t *dst = (active_buf == buf_a) ? buf_b : buf_a;
        recv_frame(dst);
        active_buf = dst;  // 原子指针赋值 (ARM 32-bit aligned store)
        watchdog_update();  // 帧成功 → 喂狗
    }
}
