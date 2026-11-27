"""Discover the real FC mode switches (AUX channels) over MSP.

The companion must never assume AUX indices from a config file: the switch
layout is a property of the flight controller.  discover_aux() reads
MSP_BOXNAMES/MSP_BOXIDS/MSP_MODE_RANGES and resolves:

* ARM          permanent ID 0
* ANGLE        permanent ID 1
* MSP OVERRIDE permanent ID 50 (BOX_MSPOVERRIDE)

Returned channel indices are 0-based into the full AETR 16-channel array used
by onboard.config.FcConfig and the RCMux (0 roll, 1 pitch, 2 throttle, 3 yaw,
4 = AUX1, ...).

The "engage" channel is not an FC mode: it is the pilot switch the companion
reads, so it comes from cfg.aux_engage when a config is supplied.  It is
reported here only so scripts/fc_identify.py can emit one coherent snippet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from msp import (
    MSP_CMD,
    decode_boxids,
    decode_boxnames,
    decode_mode_ranges,
    decode_mode_ranges_extra,
    decode_rx_map,
)

BOX_PERMANENT_ID_ARM = 0
BOX_PERMANENT_ID_ANGLE = 1
BOX_PERMANENT_ID_MSPOVERRIDE = 50
MAX_MODE_PAGES = 8
BOX_PAGE_SIZE = 32


def _query(link, cmd, payload=b"", is_v2=False):
    """Call a link query across the small signature variations test doubles use."""
    try:
        return link.query(cmd, is_v2=is_v2, payload=payload)
    except TypeError:
        pass
    try:
        return link.query(cmd, is_v2=is_v2)
    except TypeError:
        return link.query(cmd)


def _query_box_page(link, cmd, page):
    payload = bytes([page])
    for is_v2 in (False, True):
        result = _query(link, cmd, payload=payload, is_v2=is_v2)
        if result:
            return result
    return None


def _collect_pages(link, cmd):
    pages = []
    for page in range(MAX_MODE_PAGES):
        payload = _query_box_page(link, cmd, page)
        if not payload:
            break
        pages.append(payload)
        if cmd == MSP_CMD.BOXIDS and len(payload) < BOX_PAGE_SIZE:
            break
        if cmd == MSP_CMD.BOXNAMES and len(decode_boxnames(payload)) < BOX_PAGE_SIZE:
            break
    return pages


@dataclass
class AuxDiscovery:
    """Resolved mode-switch layout.  Channel values are full-array indices."""

    boxes: list = field(default_factory=list)            # [(permanent_id, name)]
    bit_index_by_pid: dict = field(default_factory=dict) # {permanent_id: BOXIDS index}
    mode_ranges: list = field(default_factory=list)      # decoded + permanent id
    mode_ranges_extra: list = field(default_factory=list) # MSP_MODE_RANGES_EXTRA (238)
    rx_map: Optional[tuple] = None                       # MSP_RX_MAP (64), AETR=(0,1,3,2)
    override_channel: Optional[int] = None
    override_range: Optional[tuple] = None
    arm_channel: Optional[int] = None
    arm_range: Optional[tuple] = None
    angle_channel: Optional[int] = None
    angle_range: Optional[tuple] = None
    engage_channel: Optional[int] = None
    warnings: list = field(default_factory=list)

    def name_for(self, permanent_id):
        for pid, name in self.boxes:
            if pid == permanent_id:
                return name
        return None

    def bit_index_for(self, permanent_id):
        """Return the MSP_STATUS flightModeFlags bit for a permanent ID.

        Betaflight's packFlightModeFlags() numbers bits by the *position of
        the permanent ID in the active MSP_BOXIDS list*, not by the permanent
        ID itself.  BOXIDS is ordered the same way, so the list index is the
        bit index.  Bit 0 is normally ARM, never ANGLE.
        """
        pid = int(permanent_id)
        if pid in self.bit_index_by_pid:
            return int(self.bit_index_by_pid[pid])
        for index, (box_pid, _name) in enumerate(self.boxes):
            if int(box_pid) == pid:
                return index
        return None

    def is_mode_active(self, flight_mode_flags, permanent_id,
                       extended_flags: bytes = b"") -> Optional[bool]:
        """Evaluate one mode bit, or None when the box is unknown."""
        bit = self.bit_index_for(permanent_id)
        if bit is None:
            return None
        if bit < 32:
            return bool(int(flight_mode_flags) & (1 << bit))
        byte_index = (bit - 32) // 8
        bit_offset = (bit - 32) % 8
        if byte_index < len(extended_flags):
            return bool(extended_flags[byte_index] & (1 << bit_offset))
        return None

    def rx_map_is_aetr(self) -> Optional[bool]:
        if self.rx_map is None:
            return None
        mapping = tuple(self.rx_map)
        if mapping[:4] != (0, 1, 3, 2):
            return False
        # Newer firmware reports all eight mappable inputs.  A, E, T, R alone
        # is not enough: a swapped AUX1..AUX4 would silently reinterpret the
        # ARM/ANGLE/OVERRIDE switches.
        if len(mapping) >= 8:
            return mapping[:8] == (0, 1, 3, 2, 4, 5, 6, 7)
        return True

    def to_dict(self):
        return {
            "boxes": [{"permanent_id": pid, "name": name} for pid, name in self.boxes],
            "bit_index_by_pid": {str(k): int(v) for k, v in self.bit_index_by_pid.items()},
            "mode_ranges": list(self.mode_ranges),
            "mode_ranges_extra": list(self.mode_ranges_extra),
            "rx_map": list(self.rx_map) if self.rx_map is not None else None,
            "channels": {
                "arm": self.arm_channel,
                "angle": self.angle_channel,
                "override": self.override_channel,
                "engage": self.engage_channel,
            },
            "override_range": list(self.override_range) if self.override_range else None,
            "arm_range": list(self.arm_range) if self.arm_range else None,
            "angle_range": list(self.angle_range) if self.angle_range else None,
            "warnings": list(self.warnings),
        }

    def config_snippet(self):
        fc = {}
        if self.arm_channel is not None:
            fc["aux_arm"] = int(self.arm_channel)
        if self.angle_channel is not None:
            fc["aux_angle"] = int(self.angle_channel)
        if self.override_channel is not None:
            fc["aux_override"] = int(self.override_channel)
        if self.engage_channel is not None:
            fc["aux_engage"] = int(self.engage_channel)
        return {"fc": fc}


def _record_ranges(discovery, payloads):
    raw = []
    for payload in payloads:
        raw.extend(decode_mode_ranges(payload))
    resolved = []
    box_ids = [pid for pid, _ in discovery.boxes]
    box_names = [name for _, name in discovery.boxes]
    name_by_pid = {pid: name for pid, name in discovery.boxes}
    for item in raw:
        if not item["valid"]:
            continue
        box_id = int(item["box_id"])
        # Current Betaflight writes the permanent ID here.  Older firmware
        # wrote the box index; fall back to that interpretation only if the
        # value is not a known permanent ID but is a valid box index.
        if box_id in name_by_pid:
            permanent_id = box_id
        elif 0 <= box_id < len(box_ids):
            permanent_id = box_ids[box_id]
        else:
            permanent_id = box_id
        entry = dict(item)
        entry["permanent_id"] = permanent_id
        entry["name"] = name_by_pid.get(permanent_id, box_names[box_id] if 0 <= box_id < len(box_names) else None)
        entry["channel_index"] = 4 + int(item["aux_index"])
        resolved.append(entry)
    discovery.mode_ranges = resolved
    return resolved


def resolve_aux_layout(ids, names=(), range_payloads=(), rx_map=None,
                       extra_payload=None, cfg=None) -> AuxDiscovery:
    """Build an AuxDiscovery from already-fetched MSP replies (pure).

    The runtime fetches these replies across several poll ticks to avoid
    blocking the telemetry thread; discovery and offline tools can also fetch
    them synchronously and call this function.  ``ids`` must preserve the
    MSP_BOXIDS page order because that order defines the status-flag bits.
    """
    discovery = AuxDiscovery()
    ids = [int(pid) for pid in ids]
    # BOXNAMES is optional for bit mapping; fill blanks so lengths stay aligned.
    names = list(names or ())
    if len(names) != len(ids):
        discovery.warnings.append(
            "BOXNAMES/BOXIDS length mismatch: %d names vs %d ids" % (len(names), len(ids))
        )
    discovery.boxes = [(pid, names[i] if i < len(names) else "")
                       for i, pid in enumerate(ids)]
    discovery.bit_index_by_pid = {pid: index for index, pid in enumerate(ids)}

    if rx_map is not None:
        discovery.rx_map = tuple(int(v) for v in rx_map)
        if not discovery.rx_map_is_aetr():
            discovery.warnings.append(
                "FC MSP_RX_MAP %s is not AETR1234 (0,1,3,2)"
                % (discovery.rx_map[:4],)
            )
    else:
        discovery.warnings.append("MSP_RX_MAP unavailable; cannot verify AETR1234")

    _record_ranges(discovery, range_payloads)

    if extra_payload:
        discovery.mode_ranges_extra = decode_mode_ranges_extra(extra_payload)
        for entry in discovery.mode_ranges_extra:
            if entry["permanent_id"] == 0 and entry["mode_logic"] == 0 and entry["linked_to"] == 0:
                continue
            if entry["mode_logic"] != 0 or entry["linked_to"] != 0:
                discovery.warnings.append(
                    "mode slot %d (pid=%d) uses logic=%d linked_to=%d; "
                    "verify the FC actually applies it"
                    % (entry["slot"], entry["permanent_id"],
                       entry["mode_logic"], entry["linked_to"])
                )

    name_by_pid = {pid: name for pid, name in discovery.boxes}
    if BOX_PERMANENT_ID_MSPOVERRIDE not in name_by_pid:
        discovery.warnings.append("BOXMSPOVERRIDE (permanent ID 50) not advertised by FC")
    if cfg is not None:
        discovery.engage_channel = int(getattr(cfg, "aux_engage", 7))

    def pick(permanent_id):
        for item in discovery.mode_ranges:
            if item["permanent_id"] == permanent_id:
                return item
        return None

    def channel_of(item):
        if item is None:
            return None
        channel = 4 + int(item["aux_index"])
        if channel > 15:
            discovery.warnings.append(
                "mode range for %s uses AUX index %d (out of 16ch range)"
                % (item.get("name"), item["aux_index"])
            )
            return None
        return channel

    def rng(item):
        return (item["start_pwm"], item["end_pwm"]) if item is not None else None

    arm = pick(BOX_PERMANENT_ID_ARM)
    angle = pick(BOX_PERMANENT_ID_ANGLE)
    override = pick(BOX_PERMANENT_ID_MSPOVERRIDE)
    discovery.arm_channel = channel_of(arm)
    discovery.arm_range = rng(arm)
    discovery.angle_channel = channel_of(angle)
    discovery.angle_range = rng(angle)
    discovery.override_channel = channel_of(override)
    discovery.override_range = rng(override)
    if override is None and BOX_PERMANENT_ID_MSPOVERRIDE in name_by_pid:
        discovery.warnings.append(
            "BOXMSPOVERRIDE box exists but has no configured mode range"
        )
    for label, item in (("ARM", arm), ("ANGLE", angle)):
        if item is None:
            discovery.warnings.append("%s mode range not configured on FC" % label)
    if discovery.engage_channel is None:
        discovery.warnings.append(
            "engage channel unknown (no FcConfig supplied; it is a pilot switch, not an FC box)"
        )
    return discovery


def discover_aux(link, cfg=None) -> AuxDiscovery:
    """Read box names/IDs/mode ranges and resolve the real AUX channels."""
    id_pages = _collect_pages(link, MSP_CMD.BOXIDS)
    ids = []
    for payload in id_pages:
        ids.extend(decode_boxids(payload))

    names = []
    for page in range(len(id_pages) or 1):
        payload = _query_box_page(link, MSP_CMD.BOXNAMES, page)
        if not payload:
            break
        names.extend(decode_boxnames(payload))

    range_pages = []
    payload = _query(link, MSP_CMD.MODE_RANGES, payload=b"", is_v2=False)
    if payload is None:
        payload = _query(link, MSP_CMD.MODE_RANGES, payload=b"", is_v2=True)
    if payload is not None:
        range_pages.append(payload)

    extra_payload = _query(link, MSP_CMD.MODE_RANGES_EXTRA, payload=b"", is_v2=False)
    if extra_payload is None:
        extra_payload = _query(link, MSP_CMD.MODE_RANGES_EXTRA, payload=b"", is_v2=True)

    rx_map_payload = _query(link, MSP_CMD.RX_MAP, payload=b"", is_v2=False)
    rx_map = decode_rx_map(rx_map_payload) if rx_map_payload else None

    discovery = resolve_aux_layout(
        ids, names, range_pages, rx_map=rx_map,
        extra_payload=extra_payload, cfg=cfg,
    )
    if not range_pages:
        discovery.warnings.append("MSP_MODE_RANGES unavailable; AUX channels unresolved")
    return discovery
