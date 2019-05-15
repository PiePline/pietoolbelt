import cv2
import numpy as np

__all__ = ['ImgMatcher']


class ImgMatcher:
    class _Img:
        def __init__(self, img):
            self.img = img
            self.kps, self.features = None, None
            self._detect_features()

        def _detect_features(self):
            descriptor = cv2.ORB_create()
            kps, self.features = descriptor.detectAndCompute(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), None)

            self.kps = np.float32([kp.pt for kp in kps])

        def mask_like(self, fill=0):
            res = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=self.img.dtype)
            if fill != 0:
                res.fill(fill)
            return res

    def __init__(self, base_img: np.ndarray, additional_img: np.ndarray):
        self._base_img = self._Img(base_img)
        self._additional_img = self._Img(additional_img)

        self._matches = None
        self._homography = None
        self._transtated_h = None

        self._warped_offset = None
        self._warped_wh = None
        self._composed_shape = None

        self._base_contour = None
        self._warped_contour = None

        self._add_img_offset = None
        self._base_img_offset = None

        self._warped_size_thresold = None

    def enable_warped_size_check(self, thresold: float) -> None:
        self._warped_size_thresold = thresold

    def compose_images(self):
        mask = self.get_overlap_mask()

        if mask is None:
            return None

        mask_inner = cv2.dilate(mask.copy(), np.ones((3, 3), dtype=np.uint8))
        mask_inner = np.expand_dims(mask_inner, axis=2)

        self._calc_translated_h()

        warped = cv2.warpPerspective(self._additional_img.img, self._transtated_h, (self._warped_wh[0], self._warped_wh[1]))
        warped_res = np.zeros((self._composed_shape[0], self._composed_shape[1], 3), dtype=self._base_img.img.dtype)
        base_res = warped_res.copy()

        warped_res[self._add_img_offset[1]: warped.shape[0] + self._add_img_offset[1], self._add_img_offset[0]: warped.shape[1] + self._add_img_offset[0]] = warped
        base_res[self._base_img_offset[1]: self._base_img.img.shape[0] + self._base_img_offset[1], self._base_img_offset[0]: self._base_img.img.shape[1] + self._base_img_offset[0]] = self._base_img.img

        res = np.where(np.concatenate([mask_inner, mask_inner, mask_inner], axis=2) > 0, base_res / 2 + warped_res / 2, base_res + warped_res).astype(np.uint8)
        return res

    def draw_matches(self):
        # initialize the output visualization image
        (hA, wA) = self._base_img.img.shape[:2]
        (hB, wB) = self._additional_img.img.shape[:2]
        res = np.zeros((max(hA, hB), wA + wB, 3), dtype=np.uint8)
        res[0: hA, 0: wA] = self._base_img.img
        res[0: hB, wA:] = self._additional_img.img

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(self._matches, self._homography[1]):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(self._base_img.kps[queryIdx][0]), int(self._base_img.kps[queryIdx][1]))
                ptB = (int(self._additional_img.kps[trainIdx][0]) + wA, int(self._additional_img.kps[trainIdx][1]))
                cv2.line(res, ptA, ptB, (0, 255, 0), 1)
        return res

    def get_overlap_area(self):
        return np.count_nonzero(self.get_overlap_mask())

    def get_overlap_mask(self):
        if self._homography is None:
            return None

        self._calc_warped_contour()

        if not self._check_warped_size():
            return None

        self._calc_base_contour()

        mask = np.zeros(self._composed_shape, dtype=np.uint8)
        warped_mask = cv2.fillPoly(mask.copy(), [self._warped_contour], 1)
        base_mask = cv2.fillPoly(mask.copy(), [self._base_contour], 1)
        mask[(warped_mask + base_mask) > 1] = 255

        return mask

    def calc_matches(self, ratio):
        matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        rawMatches = matcher.knnMatch(self._base_img.features, self._additional_img.features, 2)

        self._matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                self._matches.append((m[0].trainIdx, m[0].queryIdx))

    def calc_homography(self, reproj_thresh=1):
        if self._matches is not None and len(self._matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([self._base_img.kps[i] for (_, i) in self._matches])
            ptsB = np.float32([self._additional_img.kps[i] for (i, _) in self._matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsB, ptsA, cv2.LMEDS, reproj_thresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            self._homography = H, status

    def transform_shape(self, shape: np.ndarray) -> np.ndarray:
        """
        Transform shape from additional image coordinate system to base CS

        :param shape: shape to transform/ Array shape will be (n, 2)
        :return: transformed shape
        """
        return self._transform_shape(shape, is_translated_homography=True)

    def _transform_shape(self, shape: np.ndarray, is_translated_homography: bool) -> np.ndarray or None:
        if is_translated_homography:
            if self._transtated_h is None:
                return None
            H = self._transtated_h
            return np.squeeze(cv2.perspectiveTransform(shape.reshape(-1, 1, 2).astype(np.float32), H)) + self._add_img_offset
        else:
            if self._homography is None:
                return None
            H, _ = self._homography
            return np.squeeze(cv2.perspectiveTransform(shape.reshape(-1, 1, 2).astype(np.float32), H))

    def get_warped_poly(self):
        self._calc_translated_h()

    def _calc_translated_h(self):
        if self._transtated_h is None:
            translation = np.array([[1, 0, -self._warped_offset[0]],
                                    [0, 1, -self._warped_offset[1]],
                                    [0, 0, 1]])

            H, _ = self._homography
            self._transtated_h = translation.dot(H)

    def _check_warped_size(self) -> bool:
        if self._warped_size_thresold is not None:
            if (self._warped_wh[0] * self._warped_wh[1]) / (self._base_img.img.shape[0] * self._base_img.img.shape[1]) > self._warped_size_thresold:
                return False
        return True

    def _calc_composed_shape(self):
        if self._composed_shape is None:
            bh, bw = self._base_img.img.shape[:2]
            warped_pts = np.array([[0, 0], [0, self._warped_wh[1] - 1], [self._warped_wh[0] - 1, self._warped_wh[1] - 1], [self._warped_wh[0] - 1, 0]]) + self._warped_offset
            base_pts = np.array([[0, 0], [0, bh - 1], [bw - 1, bh - 1], [bw - 1, 0]])
            points = np.concatenate([warped_pts, base_pts], axis=0)
            self._composed_shape = np.ceil(points.max(0) - points.min(0) + np.array([1, 1]))
            self._composed_shape = (int(self._composed_shape[1]), int(self._composed_shape[0]))

            self._base_img_offset = -points.min(0)
            self._base_img_offset = np.ceil(self._base_img_offset).astype(int)
            self._add_img_offset = self._warped_offset - points.min(0)
            self._add_img_offset[self._add_img_offset < 0] = 0
            self._add_img_offset = np.ceil(self._add_img_offset).astype(int)

        return self._composed_shape

    def _calc_base_contour(self):
        if self._base_contour is None:
            bh, bw = self._base_img.img.shape[:2]
            base_pts = np.array([[0, 0], [0, bh - 1], [bw - 1, bh - 1], [bw - 1, 0]])
            self._base_contour = (base_pts + self._base_img_offset).astype(np.int)

        return self._base_contour

    def _calc_warped_contour(self):
        if self._warped_contour is None:
            h, w = self._additional_img.img.shape[:2]
            matched_box = self._transform_shape(np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]), False)
            self._warped_offset = np.min(matched_box, axis=0)
            self._warped_wh = np.max(matched_box, axis=0) - self._warped_offset

            self._calc_composed_shape()

            self._warped_contour = (matched_box - self._warped_offset + self._add_img_offset).astype(np.int)

        return self._warped_contour
