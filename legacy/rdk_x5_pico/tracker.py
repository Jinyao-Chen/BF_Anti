"""
Lightweight object tracker for RDK X5 YOLO detection pipeline.

Inspired by ByteTrack / BoT-SORT — uses IoU association + motion prediction.

Features:
  - IoU-based greedy matching (high-score first, low-score second pass)
  - Constant-velocity motion prediction (reduces 漏检率)
  - EMA box smoothing (eliminates jitter, improves 观感)
  - Track trail history for movement visualization
  - Auto-create/confirm/delete tracks
"""

import cv2
import numpy as np
from collections import deque


# --------------------------------------------------------------------------- #
# IoU helpers
# --------------------------------------------------------------------------- #

def iou(box_a, box_b):
    """IoU of two boxes, each in [x1, y1, x2, y2] format."""
    xx1 = max(box_a[0], box_b[0])
    yy1 = max(box_a[1], box_b[1])
    xx2 = min(box_a[2], box_b[2])
    yy2 = min(box_a[3], box_b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    if inter == 0:
        return 0.0
    a1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    a2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (a1 + a2 - inter)


def distance_matrix(track_preds, det_boxes):
    """用中心点距离代替 IoU 进行小目标匹配"""
    if not track_preds or not len(det_boxes):
        return np.zeros((len(track_preds), len(det_boxes)), dtype=np.float32)
    
    mat = np.zeros((len(track_preds), len(det_boxes)), dtype=np.float32)
    for i, tp in enumerate(track_preds):
        # 追踪框的中心点
        cx1, cy1 = (tp[0]+tp[2])/2, (tp[1]+tp[3])/2
        for j in range(len(det_boxes)):
            db = det_boxes[j]
            # 检测框的中心点
            cx2, cy2 = (db[0]+db[2])/2, (db[1]+db[3])/2
            
            # 计算欧氏距离
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # 距离越小，代表越可能是同一个目标 (转成类似 IoU 的分数，最大为1)
            # 这里的 100 是像素容忍度，对应高速无人机在 1280x720 下的帧间位移
            score = max(0, 1.0 - (dist / 100.0)) 
            mat[i, j] = score
    return mat


def greedy_match(iou_mat, min_iou=0.3):
    """Greedy bipartite matching by descending IoU.

    Returns:
        matches:         list of [(trk_idx, det_idx), ...]
        unmatched_trks:  list of trk_idx not matched
        unmatched_dets:  list of det_idx not matched
    """
    n_trk, n_det = iou_mat.shape
    trk_matched = [False] * n_trk
    det_matched = [False] * n_det
    matches = []

    # sort all candidate pairs by IoU descending
    pairs = [(iou_mat[i, j], i, j) for i in range(n_trk) for j in range(n_det)]
    pairs.sort(key=lambda x: -x[0])

    for val, i, j in pairs:
        if val < min_iou:
            break
        if trk_matched[i] or det_matched[j]:
            continue
        trk_matched[i] = True
        det_matched[j] = True
        matches.append((i, j))

    unmatched_trks = [i for i in range(n_trk) if not trk_matched[i]]
    unmatched_dets = [j for j in range(n_det) if not det_matched[j]]
    return matches, unmatched_trks, unmatched_dets


# --------------------------------------------------------------------------- #
# Track colour palette — one colour per track_id
# --------------------------------------------------------------------------- #

TRACK_COLORS = [
    (255, 56, 56),   (56, 255, 56),   (56, 56, 255),
    (255, 200, 56),  (200, 255, 56),  (56, 200, 255),
    (255, 100, 200), (200, 56, 255),  (56, 255, 200),
    (255, 150, 100), (150, 255, 100), (100, 150, 255),
    (255, 56, 150),  (150, 56, 255),  (56, 200, 150),
    (255, 200, 150), (200, 150, 255), (150, 200, 56),
    (255, 100, 100), (100, 255, 255),
]

def track_color(track_id):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


# --------------------------------------------------------------------------- #
# Track states
# --------------------------------------------------------------------------- #

class _State:
    TENTATIVE = 0   # newly created, needs a few hits to confirm
    CONFIRMED = 1   # stable, visible
    LOST      = 2   # gone for too many frames, will be deleted


# --------------------------------------------------------------------------- #
# Track
# --------------------------------------------------------------------------- #

class Track:
    """One tracked object — holds smoothed box, velocity, trail."""

    MAX_VELOCITY = 80.0  # max pixels/frame (fast drones)

    def __init__(self, box, score, cls_id, track_id, ema_alpha=0.65):
        self.track_id = track_id
        self.cls_id   = int(cls_id)
        self.state    = _State.TENTATIVE
        self.alpha    = ema_alpha          # EMA: 0.85=fast response, 0.65=moderate

        # smoothed box (EMA) — this is what gets shown
        self.box = np.array(box, dtype=np.float32).copy()

        # raw detection (for velocity)
        self._raw_box = np.array(box, dtype=np.float32).copy()
        self._velocity = np.zeros(4, dtype=np.float32)

        self.score     = score
        self.hit_count = 1        # frames where this track matched a detection
        self.age       = 1        # total frames alive (including predicted)
        self.lost_count = 0       # consecutive frames without match
        self.max_lost  = 30       # quit after this many lost frames

        # centre trail (for drawing movement path)
        self.trail = deque(maxlen=30)
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        self.trail.append((int(cx), int(cy)))

    # -- properties --------------------------------------------------------- #

    @property
    def is_visible(self):
        return self.state in (_State.TENTATIVE, _State.CONFIRMED)

    @property
    def is_confirmed(self):
        return self.state == _State.CONFIRMED

    @property
    def predicted_box(self):
        """Where we expect the object to be right now (used for matching).

        Note: predict() already moves self.box by velocity each frame,
        so we just return the current box — no double-count.
        """
        return self.box

    # -- update / predict --------------------------------------------------- #

    def _update_velocity(self, det_box):
        raw_vel = det_box - self._raw_box
        self._velocity = np.clip(raw_vel, -self.MAX_VELOCITY, self.MAX_VELOCITY)
        self._raw_box = det_box.copy()

    def update(self, box, score):
        """Matched a detection — update state and smooth box."""
        det_box = np.array(box, dtype=np.float32)
        # EMA: blend detection with current smoothed box
        self.box = self.alpha * det_box + (1.0 - self.alpha) * self.box
        self._update_velocity(det_box)

        self.score = score
        self.hit_count += 1
        self.age += 1
        self.lost_count = 0

        # tentative → confirmed after 3 hits
        if self.state == _State.TENTATIVE and self.hit_count >= 3:
            self.state = _State.CONFIRMED

        # trail
        cx = (self.box[0] + self.box[2]) * 0.5
        cy = (self.box[1] + self.box[3]) * 0.5
        self.trail.append((int(cx), int(cy)))

    def predict(self):
        """No detection matched — advance using velocity and clamp to prevent drift."""
        self.box += self._velocity
        self._raw_box += self._velocity     # keep raw_box in sync so velocity
        self.age += 1                       # doesn't spike on reconnection
        self.lost_count += 1

        if self.lost_count > 1:
            cx = (self.box[0] + self.box[2]) * 0.5
            cy = (self.box[1] + self.box[3]) * 0.5
            self.trail.append((int(cx), int(cy)))

        if self.lost_count >= self.max_lost:
            self.state = _State.LOST

    # -- drawing colour ----------------------------------------------------- #

    @property
    def color(self):
        return track_color(self.track_id)


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #

class Tracker:
    """Orchestrates track lifecycle per frame.

    Call order::

        tracker = Tracker(...)
        while True:
            boxes, scores, cls_ids = model.predict(frame)
            tracks = tracker.step(boxes, scores, cls_ids)
            # tracks  →  (boxes, scores, cls_ids, track_ids)
    """

    def __init__(self,
                 dist_thresh=0.15,
                 low_dist_thresh=0.05,
                 ema_alpha=0.85,
                 max_lost=10,
                 min_hits=1,
                 conf_thresh=0.05):
        """
        Args:
            dist_thresh:     Center-distance score threshold for high-conf matches
                             (1.0 = same position, 0.0 = 100px apart).
            low_dist_thresh: Distance threshold for low-confidence second pass.
            ema_alpha:       EMA smoothing factor (0..1).  1=no smoothing, 0.5=moderate.
            max_lost:        Frames to keep predicting after losing a track.
            min_hits:        Hits before tentative → confirmed.
            conf_thresh:     Minimum score to accept a detection at all.
        """
        self.dist_thresh     = dist_thresh
        self.low_dist_thresh = low_dist_thresh
        self.ema_alpha      = ema_alpha
        self.max_lost       = max_lost
        self.min_hits       = min_hits
        self.conf_thresh    = conf_thresh
        self._tracks = []       # list of Track objects
        self._next_id = 0
        self.frame_count = 0

    # -- public ------------------------------------------------------------- #

    @property
    def tracks(self):
        """All currently visible tracks."""
        return [t for t in self._tracks if t.is_visible]

    @property
    def active_tracks(self):
        """Only confirmed tracks (not tentative)."""
        return [t for t in self._tracks if t.is_confirmed]

    def step(self, boxes, scores, cls_ids):
        """Process one frame.

        Args:
            boxes:   (N, 4)  array [x1,y1,x2,y2]
            scores:  (N,)    array of confidences
            cls_ids: (N,)    array of class indices

        Returns:
            out_boxes:   list of smoothed [x1,y1,x2,y2]
            out_scores:  list of scores
            out_cls_ids: list of class IDs
            out_ids:     list of persistent track IDs
        """
        self.frame_count += 1

        # 1. split detections by score
        high_dets, low_dets = self._split(boxes, scores, cls_ids, high_thresh=0.30)

        # 2. active tracks (visible)
        active = [t for t in self._tracks if t.is_visible]

        # 3. first association: high-confidence detections → tracks
        if len(active) and len(high_dets):
            preds = [t.predicted_box for t in active]
            mat   = distance_matrix(preds, high_dets[:, :4])
            matches, un_trk, un_det = greedy_match(mat, self.dist_thresh)
            trk_high = [(active[i], high_dets[j]) for i, j in matches]
            unmatched_trks  = [active[i] for i in un_trk]
            unmatched_high  = high_dets[un_det] if len(un_det) else np.empty((0, 6))
        else:
            trk_high = []
            unmatched_trks  = active
            unmatched_high  = high_dets

        # 4. second association: low-confidence detections → remaining tracks
        #    (ByteTrack idea — catches the edge cases)
        if len(unmatched_trks) and len(low_dets):
            preds2 = [t.predicted_box for t in unmatched_trks]
            mat2   = distance_matrix(preds2, low_dets[:, :4])
            matches2, _, un_det2 = greedy_match(mat2, self.low_dist_thresh)
            # only unmatched tracks from here become truly unmatched
            trk_low = [(unmatched_trks[i], low_dets[j]) for i, j in matches2]
            truly_unmatched = [t for i, t in enumerate(unmatched_trks)
                               if i not in {m[0] for m in matches2}]
        else:
            trk_low = []
            truly_unmatched = unmatched_trks

        # 5. update matched tracks
        for t, det in trk_high + trk_low:
            t.update(det[:4], float(det[4]))

        # 6. predict unmatched tracks (keep alive with velocity)
        for t in truly_unmatched:
            t.predict()

        # 7. create new tracks from remaining high-confidence detections
        #    (only unmatched_high — don't spawn from low-score detections)
        if len(unmatched_high):
            for det in unmatched_high:
                trk = Track(det[:4], float(det[4]),
                            int(det[5]), self._next_id,
                            ema_alpha=self.ema_alpha)
                trk.max_lost = self.max_lost
                if self.min_hits <= 1:
                    trk.state = _State.CONFIRMED
                self._tracks.append(trk)
                self._next_id += 1

        # 8. garbage collect dead tracks
        self._tracks = [t for t in self._tracks if t.state != _State.LOST]

        # 9. build output
        visible = self.tracks
        if not visible:
            return [], [], [], []

        out_boxes   = [t.box.tolist() for t in visible]
        out_scores  = [t.score        for t in visible]
        out_cls_ids = [t.cls_id       for t in visible]
        out_ids     = [t.track_id     for t in visible]
        return out_boxes, out_scores, out_cls_ids, out_ids

    # -- internal ----------------------------------------------------------- #

    def _split(self, boxes, scores, cls_ids, high_thresh=0.3):
        """Split into high / low confidence detection arrays (N, 6)."""
        high, low = [], []
        for b, sc, c in zip(boxes, scores, cls_ids):
            if sc < self.conf_thresh:
                continue
            det = np.array([*b, sc, c], dtype=np.float32)
            if sc >= high_thresh:
                high.append(det)
            else:
                low.append(det)
        high = np.array(high) if high else np.empty((0, 6))
        low  = np.array(low)  if low  else np.empty((0, 6))
        return high, low

    def reset(self):
        """Clear all tracks (e.g. after changing scene)."""
        self._tracks.clear()
        self._next_id = 0
        self.frame_count = 0


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #

def draw_tracks(img, tracks, max_trail=20, thickness=2,
                label_scale=0.45, label_thickness=1, trail_width=1):
    """Draw tracked objects on a copy of the image.

    Renders per track:
      - smoothed bounding box  (thick border, confirmed only)
      - track ID + score label
      - movement trail (fading dots + connecting line)
      - tentative tracks drawn dashed

    Args:
        img:          OpenCV BGR image (modified in-place)
        tracks:       list of Track objects
        max_trail:    max trail points to draw
        thickness:    box border thickness (px)
        label_scale:  cv2.FONT_HERSHEY_SIMPLEX scale for ID text
        label_thickness: thickness of ID text
        trail_width:  width of trail connecting lines

    Returns the modified image.
    """
    # draw trails first (under boxes)
    for trk in tracks:
        if len(trk.trail) < 2:
            continue
        color = trk.color
        pts = list(trk.trail)[-max_trail:]
        # trail dots (older = smaller & dimmer)
        for i, (x, y) in enumerate(pts):
            r = max(1, int(2 + 3 * i / len(pts)))
            frac = 0.3 + 0.7 * i / len(pts)
            c = tuple(int(v * frac) for v in color)
            cv2.circle(img, (x, y), r, c, -1)
        # trail connecting line
        for i in range(len(pts) - 1):
            frac = 0.4 + 0.6 * i / len(pts)
            c = tuple(int(v * frac) for v in color)
            cv2.line(img, pts[i], pts[i + 1], c, trail_width, cv2.LINE_AA)

    # draw boxes on top
    for trk in tracks:
        color = trk.color
        x1, y1, x2, y2 = map(int, trk.box)

        if trk.is_confirmed:
            # solid rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        else:
            # dashed style for tentative tracks
            _draw_dashed(img, (x1, y1), (x2, y2), color, thickness)

        # label background
        label = f"ID{trk.track_id}  {trk.score:.2f}"
        if trk.lost_count > 0:
            # show lost count for tracks under prediction
            label += f"  -{trk.lost_count}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       label_scale, label_thickness)
        cv2.rectangle(img,
                       (x1, y1 - lh - 8),
                       (x1 + lw + 6, y1 - 2),
                       color, -1)  # filled
        cv2.putText(img, label,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale, (0, 0, 0), label_thickness, cv2.LINE_AA)

    return img


def _draw_dashed(img, pt1, pt2, color, thickness=2, dash_len=8):
    """Helper: draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    # top
    _dash_line(img, (x1, y1), (x2, y1), color, thickness, dash_len)
    # bottom
    _dash_line(img, (x1, y2), (x2, y2), color, thickness, dash_len)
    # left
    _dash_line(img, (x1, y1), (x1, y2), color, thickness, dash_len)
    # right
    _dash_line(img, (x2, y1), (x2, y2), color, thickness, dash_len)


def _dash_line(img, pt1, pt2, color, thickness, dash_len):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = max(abs(dx), abs(dy))
    if dist == 0:
        return
    steps = dist // dash_len
    if steps == 0:
        return
    for i in range(steps):
        if i % 2:
            continue
        sx = pt1[0] + dx * i // steps
        sy = pt1[1] + dy * i // steps
        ex = pt1[0] + dx * (i + 1) // steps
        ey = pt1[1] + dy * (i + 1) // steps
        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)),
                 color, thickness, cv2.LINE_AA)
