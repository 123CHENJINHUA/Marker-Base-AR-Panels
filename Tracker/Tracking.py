'''
User can select the key points and it will automatically absorb to the nearest corner according to Shi-Tomasi corner detection algorithm.
Points are tracked using optical flow.

"Press q to quit",
"Press e to remove the last tracking point",
"Press r to reset all tracking points"

'''

import cv2
import numpy as np

import cv2
import numpy as np

class Tracker:
    def __init__(self, mtx, dist, Building_Path, video_source=0):
        self.selected_point = None
        self.tracking_points = []
        self.tracking_status = []
        
        width = 1920
        height = 1080
        fps = 60
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(3, width)  #设置宽度
        self.cap.set(4, height)  #设置长度
        self.cap.set(5, fps)  
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))




        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None

        self.mtx = mtx
        self.dist = dist
        self.Building_Path = Building_Path

        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()

        cv2.namedWindow("image")
        cv2.namedWindow("features")
        cv2.setMouseCallback("image", self.click_and_crop)

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.tracking_points) < 4:
            self.selected_point = (x, y)

    def get_four_points(self):
        if len(self.tracking_points) == 4:
            return self.tracking_points
        
    def read_Building(self, file_path):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                # 将每行分割成x, y, z三个部分并转换为浮点数
                x, y, z = map(float, line.strip().split())
                # 将坐标元组添加到列表中
                points.append(np.array([[x],[y],[z]],dtype = np.float64))
        return points

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            image = frame.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features_image = frame.copy()

            corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
            corners = np.int0(corners)

            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(features_image, (x, y), 5, (255, 0, 0), -1)

            if self.selected_point is not None:
                min_dist = 25.0
                nearest_corner = None
                for corner in corners:
                    x, y = corner.ravel()
                    dist = np.sqrt((x - self.selected_point[0])**2 + (y - self.selected_point[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corner = (x, y)

                if nearest_corner is not None:
                    cv2.circle(image, nearest_corner, 5, (0, 255, 0), 2)
                    cv2.line(image, self.selected_point, nearest_corner, (0, 255, 0), 1)
                    self.tracking_points.append(np.array(nearest_corner, dtype=np.float32).reshape(1, 2))
                    self.tracking_status.append(1)
                    self.prev_gray = gray.copy()

                else:
                    cv2.circle(image, self.selected_point, 5, (0, 255, 0), 2)
                    self.tracking_points.append(np.array(self.selected_point, dtype=np.float32).reshape(1, 2))
                    self.tracking_status.append(0)
                    self.prev_gray = gray.copy()

                self.selected_point = None

            elif len(self.tracking_points) > 0:
                self.tracking_points = np.array(self.tracking_points, dtype=np.float32).reshape(-1, 1, 2)
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.tracking_points, None, **self.lk_params)
                good_new = next_pts[status == 1]
                good_old = self.tracking_points[status == 1]
                self.tracking_status = [self.tracking_status[i] for i in range(len(status)) if status[i] == 1]


                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    if self.tracking_status[i] == 1:
                        cv2.circle(image, (int(a), int(b)), 5, (255, 255, 255), 2)
                    else:
                        cv2.circle(image, (int(a), int(b)), 5, (0, 0, 255), 2)
                    cv2.line(image, (int(c), int(d)), (int(a), int(b)), (255, 255, 0), 1)

                self.tracking_points = good_new.reshape(-1, 1, 2)
                self.prev_gray = gray.copy()
                self.tracking_points = self.tracking_points.tolist()

                if len(self.tracking_points) >= 4:
                    object_points = np.array(self.read_Building(self.Building_Path), dtype=np.float32).reshape(-1, 3)
                    image_points = np.array(self.tracking_points, dtype=np.float32)
                    _, rvec, tvec = cv2.solvePnP(object_points, image_points, self.mtx, self.dist)
                    cv2.drawFrameAxes(image,self.mtx, self.dist, rvec, tvec, length=0.05, thickness=2)


            text = "Number of tracking points: {}".format(len(self.tracking_points))
            text_lines = [
                "Press q to quit",
                "Press e to remove the last tracking point",
                "Press r to reset all tracking points"
            ]

            cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y0, dy = 40, 20
            for i, line in enumerate(text_lines):
                y = y0 + i * dy
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("image", image)
            cv2.imshow("features", features_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("e") and len(self.tracking_points) > 0:
                self.tracking_points.pop()
            elif key == ord("r"):
                self.tracking_points = []

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cv_file = cv2.FileStorage("./charuco_camera_calibration.yaml", cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("camera_matrix").mat()
    dist = cv_file.getNode("dist_coeff").mat()
    Building_Path = "./Building.txt"
    tracker = Tracker(mtx, dist, Building_Path)
    tracker.run()