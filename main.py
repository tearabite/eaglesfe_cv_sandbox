import math
import time
from collections import deque
import cv2
import imutils
import numpy as np


def nothing(c):
    pass


class CVPipeline:
    red_lower_1 = (0, 120, 70)
    red_upper_1 = (10, 255, 255)

    red_lower_2 = (170, 120, 70)
    red_upper_2 = (180, 255, 255)

    blue_lower = (80, 85, 100)
    blue_upper = (115, 255, 255)

    text_color = (255, 255, 255)  # White
    vis_color = (0, 255, 0)  # Green
    contrails_color = (255, 0, 0)  # Blue
    pts = deque(maxlen=64)

    wnd = "Controls"
    blur = 31
    iterations = 2
    scale = 100
    epsilon = 10
    visualization = 0
    steps = 1
    fps = 0

    def __init__(self):
        self.create_trackbars()

    @staticmethod
    def get_mask_for_color(img, ranges, iterations):
        result_mask = None
        for pair in ranges:
            mask = cv2.inRange(img, pair[0], pair[1])
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), (2,2)), iterations=iterations)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), (2,2)), iterations=iterations)
            result_mask = mask if result_mask is None else cv2.add(result_mask, mask)

        return result_mask

    def create_trackbars(self):
        cv2.namedWindow(self.wnd, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Blur", self.wnd, 31, 100, nothing)
        cv2.createTrackbar("Iterations", self.wnd, 2, 10, nothing)
        cv2.createTrackbar("Scale", self.wnd, 100, 200, nothing)
        cv2.createTrackbar("Visualization", self.wnd, 0, 5, nothing)
        cv2.createTrackbar("Steps", self.wnd, 1, 4, nothing)

    def update_trackbar_values(self):
        blur = cv2.getTrackbarPos("Blur", self.wnd)
        if blur % 2 == 0:
            self.blur = (blur + 1, blur + 1)
        else:
            self.blur = (blur, blur)
        self.iterations = max(1, cv2.getTrackbarPos("Iterations", self.wnd))
        self.scale = max(0.05, cv2.getTrackbarPos("Scale", self.wnd))
        self.visualization = cv2.getTrackbarPos("Visualization", self.wnd)
        self.steps = max(1, cv2.getTrackbarPos("Steps", self.wnd))

    def loop(self):
        vid = cv2.VideoCapture(0)
        cv2.namedWindow(self.wnd, cv2.WINDOW_NORMAL)

        self.fps = 0
        frame_count = 0
        last_rate_update = None
        while True:
            # frame = cv2.imread("./red_goal.jpg")
            self.update_trackbar_values()
            ret, frame = vid.read()
            frame = self.process_frame(frame)
            frame_count += 1
            if last_rate_update is None or time.time() > last_rate_update + 1:
                self.fps = frame_count
                frame_count = 0
                last_rate_update = time.time()
            cv2.imshow('Result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        dim = (int(frame.shape[1] * self.scale / 100), int(frame.shape[0] * self.scale / 100))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(frame, self.blur, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        red_mask = self.get_mask_for_color(hsv, [(self.red_lower_1, self.red_upper_1), (self.red_lower_2, self.red_upper_2)], self.iterations)
        blue_mask = self.get_mask_for_color(hsv, [(self.blue_lower, self.blue_upper)], self.iterations)
        combined_mask = cv2.add(red_mask, blue_mask)

        contours = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        center = None

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            mask = np.zeros(combined_mask.shape, np.uint8)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            mean_color = cv2.mean(frame, mask)
            if mean_color[0] < self.blue_lower[0]:
                color = "Red"
            else:
                color = "Blue"

            name = ""
            if area > 1000 * (self.scale / 100):
                # visColor = mean_color
                if self.visualization == 0:
                    name = "Raw Contour"
                    cv2.drawContours(frame, [np.int0(largest_contour)], 0, self.vis_color, 2)
                elif self.visualization == 1:
                    name = "Approximation"
                    epsilon = cv2.arcLength(largest_contour, True) * 0.02
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    if approx.size == 4:
                        cv2.drawContours(frame, [approx], 0, self.vis_color, 2)
                        area = cv2.contourArea(approx)
                elif self.visualization == 2:
                    name = "Convex Hull"
                    hull = cv2.convexHull(largest_contour)
                    cv2.drawContours(frame, [hull], 0, self.vis_color, 2)
                    area = cv2.contourArea(hull)
                elif self.visualization == 3:
                    name = "Bounding Circle"
                    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                    cv2.circle(frame, (int(x), int(y)), int(radius), self.vis_color, 2)
                    area = math.pi * math.pow(radius, 2)
                elif self.visualization == 4:
                    name = "Bounding Rectangle"
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.vis_color, 2)
                    area = w * h
                elif self.visualization == 5:
                    name = "Rotated Rectangle"
                    rect = cv2.minAreaRect(largest_contour)
                    x, y = rect[0]
                    w, h = rect[1]
                    angle = rect[2]
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.putText(frame, "{0}deg".format(int(angle * -1)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2, cv2.LINE_AA)
                    cv2.drawContours(frame, [box], 0, self.vis_color, 2)
                    area = w * h

                # Center Dot
                moment = cv2.moments(largest_contour)
                center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                # Area Text
                area_percentage = (area / (dim[0] * dim[1])) * 100
                cv2.putText(frame, "Area: {:#.1f}%".format(area_percentage), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.text_color, 2, cv2.LINE_AA)

                # Center Text
                cv2.putText(frame, "Center: X: {:.1f}, Y: {:.1f}".format(((center[0] / dim[0]) * 100) - 50, ((center[1] / dim[1]) * -100) + 50), (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.text_color, 2, cv2.LINE_AA)

                # Visualization Name
                cv2.putText(frame, name, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.text_color, 2, cv2.LINE_AA)

                # Detected Color
                cv2.putText(frame, "Color: {0}".format(color), (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.text_color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (0, dim[1] - 25), (dim[0], dim[1]), mean_color, -1)

        self.pts.appendleft(center)

        # Contrails
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], self.contrails_color, thickness)

        # Combine multiple Mats (the various steps)
        result_frame = cv2.hconcat([frame, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), blurred, hsv][:self.steps])

        # Frame rate
        cv2.putText(result_frame, "FPS: {fps}".format(fps=self.fps), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2, cv2.LINE_AA)

        return result_frame


if __name__ == "__main__":
    pipeline = CVPipeline()
    pipeline.loop()
