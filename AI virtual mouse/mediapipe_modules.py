import cv2
import mediapipe as mp

class Handtracking:
    def __init__(self,static_image_mode=False,max_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode,max_hands,min_detection_confidence,min_tracking_confidence)
    def findhands(self,img,draw=True):
        h,w,c = img.shape
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for id,hand_lm in enumerate(hand_landmarks.landmark):
                    cx , cy = int(hand_lm.x*w) ,int(hand_lm.y*h)
                    self.lmlist.append([id,cx,cy])
                if draw:
                    self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return self.lmlist
    def detecthand(self):
        if (self.lmlist):
            if(self.lmlist[1][1] < self.lmlist[17][1]):
                self.hand = "right hand"
                return self.hand
            else :
                self.hand = "left hand"
                return self.hand
    def fingersup(self):
        self.up = [0,0,0,0,0]
        self.detecthand()
        if (self.lmlist):
            if(self.lmlist[4][1] > self.lmlist[3][1] and self.hand == "left hand"):
                self.up[0] = 1
            elif(self.lmlist[4][1] < self.lmlist[3][1] and self.hand == "right hand"):
                self.up[0] = 1
            if(self.lmlist[8][2] < self.lmlist[6][2]):
                self.up[1] = 1
            if(self.lmlist[12][2] < self.lmlist[10][2]):
                self.up[2] = 1
            if(self.lmlist[16][2] < self.lmlist[14][2]):
                self.up[3] = 1
            if(self.lmlist[20][2] < self.lmlist[17][2]):
                self.up[4] = 1
        return self.up
class posetracking:
    def __init__(self,static_image_mode=False,upperbody=False,smooth_landmark=True,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode,upperbody,smooth_landmark,min_detection_confidence,min_tracking_confidence)
    def findposes(self,img,draw=True):
        h,w,c = img.shape
        self.results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.lmlist = []
        if self.results.pose_landmarks:
            for id,pose_lm in enumerate(self.results.pose_landmarks.landmark):
                    cx , cy = int(pose_lm.x*w) ,int(pose_lm.y*h)
                    self.lmlist.append([id,cx,cy])
            if draw:
                self.mp_drawing.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return self.lmlist
class Facedetection:
    def __init__(self,min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence)
    def findfaces(self,img,draw=True):
        self.results = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        height,width,c = img.shape
        self.lmlist = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                x,y,w,h =int(detection.location_data.relative_bounding_box.xmin *width),int(detection.location_data.relative_bounding_box.ymin*height),\
                int(detection.location_data.relative_bounding_box.width*width),int(detection.location_data.relative_bounding_box.height*height)
                self.lmlist.append(x,y,w,h)
                if draw:
                    cv2.rectangle(img,(x,y-20),(x+w,y+h),(0,0,0))
        return self.lmlist
class Facemesh:
    def __init__(self,static_image_mode=False,max_faces=2,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawspec = self.mp_drawing.DrawingSpec(thickness=1,circle_radius=2)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode,max_faces,min_detection_confidence,min_tracking_confidence)
    def findpoints(self,img,draw=True):
        self.results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        height,width,c = img.shape
        self.lmlist = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                for id,lm in enumerate(facelms.landmark):
                    # print(lm.landmark)
                    cx , cy = int(lm.x*width) ,int(lm.y*height)
                    self.lmlist.append([id,cx , cy])
                if draw:
                    self.mp_drawing.draw_landmarks(img, facelms, self.mp_face_mesh.FACE_CONNECTIONS,\
                    self.drawspec,self.drawspec)

        return self.lmlist
