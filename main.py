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

    wnd = "Result"
    blur = 20
    iterations = 2
    scale = 100
    epsilon = 10
    visualization = 0
    steps = 0
    fps = 0
    zoom = 0

    ps_mask = None

    def __init__(self):
        self.create_trackbars()

    @staticmethod
    def get_mask_for_color(img, ranges, iterations):
        kernel_size = 3
        result_mask = None
        for pair in ranges:
            mask = cv2.inRange(img, pair[0], pair[1])
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size), (int(kernel_size/2), int(kernel_size/2))), iterations=iterations)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size), (int(kernel_size/2), int(kernel_size/2))), iterations=iterations)
            result_mask = mask if result_mask is None else cv2.add(result_mask, mask)

        return result_mask

    def create_trackbars(self):
        cv2.namedWindow(self.wnd, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Red Threshold", self.wnd, 0, 180, nothing)
        cv2.createTrackbar("Red Threshold Window", self.wnd, 20, 60, nothing)

        cv2.createTrackbar("Blur", self.wnd, self.blur, 100, nothing)
        cv2.createTrackbar("Iterations", self.wnd, self.iterations, 10, nothing)
        cv2.createTrackbar("Scale", self.wnd, self.scale, 200, nothing)
        cv2.createTrackbar("Visualization", self.wnd, self.visualization, 5, nothing)
        cv2.createTrackbar("Steps", self.wnd, self.steps, 4, nothing)
        cv2.createTrackbar("Zoom", self.wnd, self.zoom, 4, nothing)

    def update_trackbar_values(self):
        r_thresh = cv2.getTrackbarPos("Red Threshold", self.wnd)
        r_window = cv2.getTrackbarPos("Red Threshold Window", self.wnd)
        lower = r_thresh - r_window / 2
        lower_safe = max(0, lower)
        upper = r_thresh + r_window / 2
        upper_safe = min(180, upper)

        self.red_lower_1 = (int(lower_safe), self.red_lower_1[1], self.red_lower_1[2])
        self.red_upper_1 = (int(upper_safe), self.red_upper_1[1], self.red_upper_1[2])

        if lower < 0:
            lower = 180 + lower
            upper = 180
        elif upper > 180:
            upper = upper - 180
            lower = 0

        self.red_lower_2 = (int(lower), self.red_lower_2[1], self.red_lower_2[2])
        self.red_upper_2 = (int(upper), self.red_upper_2[1], self.red_upper_2[2])

        blur = cv2.getTrackbarPos("Blur", self.wnd)
        if blur % 2 == 0:
            self.blur = (blur + 1, blur + 1)
        else:
            self.blur = (blur, blur)
        self.iterations = max(1, cv2.getTrackbarPos("Iterations", self.wnd))
        self.scale = max(0.05, cv2.getTrackbarPos("Scale", self.wnd))
        self.visualization = cv2.getTrackbarPos("Visualization", self.wnd)
        self.steps = max(0, cv2.getTrackbarPos("Steps", self.wnd))
        self.zoom = cv2.getTrackbarPos("Zoom", self.wnd) / 10

    def loop(self):
        use_still_image = False
        if not use_still_image:
            vid = cv2.VideoCapture('/Users/scott/Developer/eaglesfe_cv_sandbox/IMG_1189.mov')

        cv2.namedWindow(self.wnd, cv2.WINDOW_NORMAL)

        self.fps = 0
        frame_count = 0
        last_rate_update = None
        while True:
            # Update the values of the adjustment sliders
            self.update_trackbar_values()

            # Get a single frame
            if use_still_image:
                frame = cv2.imread("./red_goal2.jpg")
            else:
                ret, frame = vid.read()

            if frame is None:
                vid = cv2.VideoCapture('/Users/scott/Developer/eaglesfe_cv_sandbox/IMG_1189.mov')
                continue

            frame = self.process_frame(frame)

            # Update framerate calculation
            frame_count += 1
            if last_rate_update is None or time.time() > last_rate_update + 1:
                self.fps = frame_count
                frame_count = 0
                last_rate_update = time.time()

            # Update the image on screen
            cv2.imshow(self.wnd, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not use_still_image:
            vid.release()
        cv2.destroyAllWindows()

    def get_powershot_contours(self, hsv_frame, goal_contour):

        # Goal Header Dimensions: 24x15.5
        # PowerShot Offset      : -23x-11
        # PowerShot Dimensions  : 20x8
        goal_dimensions_in = (24, 15.5)
        powershot_offset_in = (-23, -11)
        powershot_dimensions_in = (20, 8)

        gc = CVPipeline.get_contour_center(goal_contour)
        gx, gy, gw, gh = cv2.boundingRect(goal_contour)
        goal_unit = gw / goal_dimensions_in[0]

        powershot_center = (gc[0] + goal_unit * powershot_offset_in[0], gc[1] - goal_unit * powershot_offset_in[1])
        powershot_tl = (int(powershot_center[0] - goal_unit * (powershot_dimensions_in[0] / 2)), int(powershot_center[1] - goal_unit * (powershot_dimensions_in[1] / 2)))
        powershot_br = (int(powershot_center[0] + goal_unit * (powershot_dimensions_in[0] / 2)), int(powershot_center[1] + goal_unit * (powershot_dimensions_in[1] / 4)))

        mask = self.get_mask_for_color(hsv_frame, [(self.red_lower_1, self.red_upper_1), (self.red_lower_2, self.red_upper_2)], 0)
        ps_mask = np.zeros(mask.shape, np.uint8)
        cv2.rectangle(ps_mask, powershot_tl, powershot_br, (255, 255, 255), -1)
        cv2.bitwise_and(mask, ps_mask, ps_mask)

        ps_contours = cv2.findContours(ps_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ps_contours = imutils.grab_contours(ps_contours)
        ps_contours = sorted(ps_contours, key=cv2.contourArea, reverse=True)[:3]

        self.ps_mask = ps_mask
        return ps_contours

    @staticmethod
    def get_contour_center(contour):
        moment = cv2.moments(contour)
        if moment["m00"] == 0:
            return None

        return int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"])

    def process_frame(self, frame):
        # Resize image
        ratio = frame.shape[1] / frame.shape[0]
        width = int(frame.shape[1] * (self.scale / 100))
        dim = (width, int(width / ratio))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # Crop image
        start_row, start_col = int(width * self.zoom), int(dim[1] * self.zoom)
        end_row, end_col = int(width * (1 - self.zoom)), int(dim[0] * (1 - self.zoom))

        # start_row and start_col are the cordinates
        # from where we will start cropping
        # end_row and end_col is the end coordinates
        # where we stop
        frame = frame[start_row:end_row, start_col:end_col]

        # Blur image
        blurred = cv2.GaussianBlur(frame, self.blur, 0)

        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Create masks
        red_mask = self.get_mask_for_color(hsv, [(self.red_lower_1, self.red_upper_1), (self.red_lower_2, self.red_upper_2)], self.iterations)
        blue_mask = self.get_mask_for_color(hsv, [(self.blue_lower, self.blue_upper)], self.iterations)
        combined_mask = cv2.add(red_mask, blue_mask)
        ignore_mask = np.zeros(combined_mask.shape, np.uint8)
        cv2.rectangle(ignore_mask, (0, 0), (dim[0], int(dim[1] / 2)), (255, 255, 255), -1)
        cv2.bitwise_and(ignore_mask, combined_mask, combined_mask)

        # Detect contours
        contours = cv2.findContours(combined_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        hulls = np.zeros(combined_mask.shape, np.uint8)
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(hulls, [hull], 0, (255, 255, 255), -1)

        contours = cv2.findContours(hulls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        def foo(contour):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rect_area = w*h
            if (area < dim[0] * dim[1] * 0.05):
                return -100
            if h == 0:
                return -100

            return float(area)/rect_area - abs(1.5 - w/h)

        goal_contour = None
        if len(contours) > 0:
            goal_contour = max(contours, key=foo)

        if goal_contour is not None and cv2.contourArea(goal_contour) > 0.05 * dim[0] * dim[1] and CVPipeline.is_entirely_in_frame(frame, goal_contour):
            goal_center = CVPipeline.get_contour_center(goal_contour)
            if goal_center is None:
                return frame
            if goal_center[1] > frame.shape[1] / 2:
                return frame

            self.draw_goal(frame, goal_contour)
            self.draw_powershot(frame, self.get_powershot_contours(hsv, goal_contour))

        # Combine multiple Mats (the various steps)
        result_frame = [frame, self.ps_mask, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), blurred, hsv][self.steps]

        # Frame rate
        cv2.putText(result_frame, "FPS: {fps}".format(fps=self.fps), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2, cv2.LINE_AA)

        return result_frame

    def draw_powershot(self, frame, powershot_contours):
        for ps in powershot_contours:
            c = CVPipeline.get_contour_center(ps)
            if c is None:
                continue
            else:
                x = c[0]
                y = c[1]
            _, _, w, h = cv2.boundingRect(ps)
            scale = self.scale / 100
            bottom = y + int(h/2)
            ratio = w/h

            cv2.arrowedLine(frame, (x, bottom), (x, bottom - int(100 * scale)), (0, 255, 0), int(8 * scale))

    def draw_goal(self, frame, goal_contour):
        dim = frame.shape
        goal_center = CVPipeline.get_contour_center(goal_contour)
        goal_area = cv2.contourArea(goal_contour)
        goal_color = CVPipeline.get_mean_color(frame, goal_contour)
        if goal_area > 500 * (self.scale / 100):
            if self.visualization == 0:
                name = "Raw Contour"
                cv2.drawContours(frame, [np.int0(goal_contour)], 0, self.vis_color, 2)
            elif self.visualization == 1:
                name = "Approximation"
                epsilon = cv2.arcLength(goal_contour, True) * 0.02
                approx = cv2.approxPolyDP(goal_contour, epsilon, True)
                if approx.size == 4:
                    cv2.drawContours(frame, [approx], 0, self.vis_color, 2)
                    goal_area = cv2.contourArea(approx)
            elif self.visualization == 2:
                name = "Convex Hull"
                hull = cv2.convexHull(goal_contour)
                cv2.drawContours(frame, [hull], 0, self.vis_color, 2)
                goal_area = cv2.contourArea(hull)
            elif self.visualization == 3:
                name = "Bounding Circle"
                ((x, y), radius) = cv2.minEnclosingCircle(goal_contour)
                cv2.circle(frame, (int(x), int(y)), int(radius), self.vis_color, 2)
                goal_area = math.pi * math.pow(radius, 2)
            elif self.visualization == 4:
                name = "Bounding Rectangle"
                x, y, w, h = cv2.boundingRect(goal_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.vis_color, 2)
                goal_area = w * h
            elif self.visualization == 5:
                name = "Rotated Rectangle"
                rect = cv2.minAreaRect(goal_contour)
                x, y = rect[0]
                w, h = rect[1]
                angle = rect[2]
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.putText(frame, "{0}deg".format(int(angle * -1)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self.text_color, 2, cv2.LINE_AA)
                cv2.drawContours(frame, [box], 0, self.vis_color, 2)
                goal_area = w * h

            # Center Dot
            moment = cv2.moments(goal_contour)
            goal_center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
            cv2.circle(frame, goal_center, 5, (0, 255, 0), -1)

            # Area Text
            area_percentage = (goal_area / (dim[0] * dim[1])) * 100
            cv2.putText(frame, "Area: {:#.1f}%".format(area_percentage), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        self.text_color, 2, cv2.LINE_AA)

            # Center Text
            cv2.putText(frame, "Center: X: {:.1f}, Y: {:.1f}".format(((goal_center[0] / dim[0]) * 100) - 50,
                                                                     ((goal_center[1] / dim[1]) * -100) + 50), (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.text_color, 2, cv2.LINE_AA)

            # Visualization Name
            cv2.putText(frame, name, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.text_color, 2, cv2.LINE_AA)

            cv2.rectangle(frame, (0, dim[1] - 25), (dim[0], dim[1]), goal_color, -1)
        return goal_center

    @staticmethod
    def is_entirely_in_frame(frame, cnt):
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        margin = 0.00
        vm = int(frame.shape[0] * margin)
        hm = int(frame.shape[1] * margin)
        return leftmost[0] > hm and rightmost[0] < frame.shape[1] - hm and topmost[1] > vm and bottommost[1] < frame.shape[0] - vm


    @staticmethod
    def get_mean_color(frame, contour):
        mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mean_color = cv2.mean(frame, mask)
        return mean_color


if __name__ == "__main__":
    pipeline = CVPipeline()
    pipeline.loop()
