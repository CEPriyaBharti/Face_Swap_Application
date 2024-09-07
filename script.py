import cv2
import numpy as np
import dlib

# Load pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_landmarks(image, face):
    landmarks = predictor(image, face)
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

def get_triangles(landmarks):
    rect = cv2.boundingRect(landmarks)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks.tolist())
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indices = []
    for triangle in triangles:
        pts = []
        for i in range(0, 6, 2):
            pt = (triangle[i], triangle[i+1])
            for j, landmark in enumerate(landmarks):
                if pt == tuple(landmark):
                    pts.append(j)
        if len(pts) == 3:
            indices.append(pts)
    return indices

def warp_triangle(img1, img2, t1, t2):
    # Find bounding box
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []
    for i in range(3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpAffine to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2Rect = cv2.warpAffine(img1Rect, cv2.getAffineTransform(np.float32(t1Rect), np.float32(t2Rect)), size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img2Rect = img2Rect * mask

    # Convert img2Rect back to uint8
    img2Rect = np.clip(img2Rect, 0, 255).astype(np.uint8)

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1 - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] += img2Rect

def swap_faces(source_img, destination_img):
    source_faces = detector(source_img)
    destination_faces = detector(destination_img)

    if len(source_faces) == 0 or len(destination_faces) == 0:
        print("No faces detected in one of the images.")
        return destination_img

    source_landmarks = extract_landmarks(source_img, source_faces[0])
    destination_landmarks = extract_landmarks(destination_img, destination_faces[0])

    # Get triangles for warping
    triangles = get_triangles(destination_landmarks)

    warped_face = np.copy(destination_img)

    for triangle in triangles:
        t1 = [source_landmarks[i] for i in triangle]
        t2 = [destination_landmarks[i] for i in triangle]
        warp_triangle(source_img, warped_face, t1, t2)

    # Create the mask with facial landmarks and blend it better with Gaussian blur
    mask = apply_mask(destination_img, destination_landmarks)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)

    # Adjust center to move face 10 pixels downwards
    center = (destination_faces[0].center().x, destination_faces[0].center().y + 10)

    # Use seamless cloning for better blending
    output_image = cv2.seamlessClone(warped_face, destination_img, mask, center, cv2.NORMAL_CLONE)

    return output_image


def apply_mask(image, landmarks):
    mask = np.zeros_like(image)
    
    # Create a convex hull around the facial landmarks, focusing only on the facial region
    hull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))

    return mask

# Load source and destination images
source_img = cv2.imread("one.png")
destination_img = cv2.imread("two.png")

# Perform face swapping (Source to Destination)
swapped_image_1 = swap_faces(source_img, destination_img)

# Perform face swapping (Destination to Source)
swapped_image_2 = swap_faces(destination_img, source_img)

# Save and display the results
cv2.imwrite("swapped_face_1.jpg", swapped_image_1)
cv2.imwrite("swapped_face_2.jpg", swapped_image_2)

cv2.imshow("Swapped Face 1 (Source to Destination)", swapped_image_1)
cv2.imshow("Swapped Face 2 (Destination to Source)", swapped_image_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
