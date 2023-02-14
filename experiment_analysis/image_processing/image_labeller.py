import cv2
import numpy as np

class ImageLabeller:
    def __init__(self, image, image_number):
        self.img = image
        self.display = self.img.copy()
        self.points = []
        self.colors = []
        self.color_names = []
        self.image_number = image_number
    
    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.display, (x, y), 11, (0, 0, 255), -1)
    
    def save_image(self, file_path):
        cv2.imwrite(file_path + '/color_detected_image_manual{}.jpg'.format(self.image_number), self.img)

    def classify_points(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        
        while True:
            cv2.imshow("Image", self.display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        for i, point in enumerate(self.points):
            print("Point", i + 1, ":", point)
            color = input("Classify point as brown, blue, red, green, purple, pink, grey, light-pink, orange: ")
            if color == "brown":
                self.colors.append((1, 1, 47))
                self.color_names.append("brown")
            elif color == "blue":
                self.colors.append((103, 62, 0))
                self.color_names.append("blue")
            elif color == "red":
                self.colors.append((0, 1, 128))
                self.color_names.append("red")
            elif color == "green":
                self.colors.append((17, 84, 1))
                self.color_names.append("green")
            elif color == "purple":
                self.colors.append((98, 42, 89))
                self.color_names.append("purple")
            elif color == "pink":
                self.colors.append((85, 0, 163))
                self.color_names.append("pink")
            elif color == "grey":
                self.colors.append((77, 79, 73))
                self.color_names.append("grey")
            elif color == "light-pink":
                self.colors.append((130, 103, 176))
                self.color_names.append("light-pink")
            elif color == "orange":
                self.colors.append((1, 120, 199))
                self.color_names.append("orange")
            elif color == "yellow":
                self.colors.append((0, 255, 255))
                self.color_names.append("yellow")
            else:
                print("Invalid color")

        for point, color, name in zip(self.points, self.colors, self.color_names):
            cv2.circle(self.img, point, 11, color, -1)
            cv2.putText(self.img, name, (point[0], point[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # cv2.imshow("Result", self.img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# labeler = ImageLabeller("data/scanned/dtu_round_1_images_scanned/scanned_image_no0.jpg", 0)
# labeler.classify_points()