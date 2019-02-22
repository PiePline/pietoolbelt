import cv2
import numpy as np

__all__ = ['balance_sequence']


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

    def compose_images(self):
        if self._homography is None:
            return None

        h, w = self._additional_img.img.shape[0], self._base_img.img.shape[1]
        matched_box = self.transform_shape(np.array([[0, 0], [0, h], [w, h], [w, 0]]))

        shape = matched_box.max(0) - matched_box.min(0)

        H, status = self._homography
        warped = cv2.warpPerspective(self._additional_img.img, H, (int(np.ceil(shape[0])), int(np.ceil(shape[1]))))

        h, w = self._additional_img.img.shape[1], self._base_img.img.shape[0]
        shape = np.array([h, w], dtype=np.float64)
        min, max = matched_box.min(0), matched_box.max(0)
        min[min > 0] = 0
        max[0] = max[0] - h if max[0] > h else 0
        max[1] = max[1] - w if max[1] > w else 0
        shape -= min
        shape += max

        result = np.zeros((int(np.ceil(shape[1])), int(np.ceil(shape[0])), 3), dtype=self._base_img.img.dtype)
        result[0: self._base_img.img.shape[1], 0: self._base_img.img.shape[0]] = self._base_img.img
        result[0: warped.shape[1], 0: warped.shape[0]] = warped

        # result = cv2.addWeighted(result, 0.5, warped, 0.5, 0)

        return warped

    def draw_matches(self):
        # initialize the output visualization image
        (hA, wA) = self._base_img.img.shape[:2]
        (hB, wB) = self._additional_img.img.shape[:2]
        res = np.zeros((max(hA, hB), wA + wB, 3), dtype=np.uint8)
        res[0:hA, 0:wA] = self._base_img.img
        res[0:hB, wA:] = self._additional_img.img

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(self._matches, self._homography[1]):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(self._base_img.kps[queryIdx][0]), int(self._base_img.kps[queryIdx][1]))
                ptB = (int(self._additional_img.kps[trainIdx][0]) + wA, int(self._additional_img.kps[trainIdx][1]))
                cv2.line(res, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return res

    def get_overlap_area(self):
        return np.count_nonzero(self.get_overlap_mask())

    def get_overlap_mask(self):
        if self._homography is None:
            return None

        H, status = self._homography
        base_mask, additional_mask = self._base_img.mask_like(1), self._additional_img.mask_like(0)
        result = cv2.warpPerspective(additional_mask, H, (base_mask.shape[1] + additional_mask.shape[1], base_mask.shape[0]))
        result[0: additional_mask.shape[0], 0: additional_mask.shape[1]] = additional_mask

        return result

    def calc_matches(self, ratio):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
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
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            self._homography = H, status

    def transform_shape(self, shape: np.ndarray) -> np.ndarray:
        """
        Transform shape from additional image coordinate system to base CS

        :param shape: shape to transform/ Array shape will be (n, 2)
        :return: transformed shape
        """
        if self._homography is None:
            return None

        H, status = self._homography
        return np.squeeze(cv2.perspectiveTransform(shape.reshape(-1, 1, 2).astype(np.float32), H))

