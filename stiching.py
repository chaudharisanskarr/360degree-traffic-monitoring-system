import cv2
import numpy as np
import os

def stitch_images(image1, image2, image3):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    # Use ORB to detect keypoints and extract descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    kp3, des3 = orb.detectAndCompute(gray3, None)

    # Match keypoints between images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches12 = bf.match(des1, des2)
    matches13 = bf.match(des1, des3)

    # Sort matches by distance
    matches12 = sorted(matches12, key=lambda x: x.distance)
    matches13 = sorted(matches13, key=lambda x: x.distance)

    # Estimate homographies
    src_pts1 = np.float32([kp1[m.queryIdx].pt for m in matches12]).reshape(-1, 1, 2)
    dst_pts1 = np.float32([kp2[m.trainIdx].pt for m in matches12]).reshape(-1, 1, 2)
    M12, mask12 = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)

    src_pts2 = np.float32([kp1[m.queryIdx].pt for m in matches13]).reshape(-1, 1, 2)
    dst_pts2 = np.float32([kp3[m.trainIdx].pt for m in matches13]).reshape(-1, 1, 2)
    M13, mask13 = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)

    # Warp image1 to image2 and image3
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h3, w3 = image3.shape[:2]
    warped_image1 = cv2.warpPerspective(image1, M12, (w2, h2))
    warped_image2 = cv2.warpPerspective(image1, M13, (w3, h3))

    # Combine image2 and warped_image1
    result = np.where(warped_image1 == 0, image2, warped_image1)
    # Combine image3 and warped_image2
    result = np.where(warped_image2 == 0, image3, warped_image2)

    return result

def main():
    # Read images
    image1 = cv2.imread('Panaroma/video_1/transformed_frame_0000.jpg')
    image2 = cv2.imread('Panaroma/video_2/transformed_frame_0000.jpg')
    image3 = cv2.imread('Panaroma/video_3/transformed_frame_0000.jpg')

    # Stitch images together
    result = stitch_images(image1, image2, image3)

    # Create output folder if it doesn't exist
    output_folder = 'Stitched_Images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Display or save stitched image
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save stitched image
    cv2.imwrite(os.path.join(output_folder, 'stitched_image.jpg'), result)
    print("Stitched image saved in 'output' folder.")

if __name__ == "__main__":
    main()
