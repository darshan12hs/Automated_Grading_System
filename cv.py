import cv2
import numpy as np
import utils

def grade_test_image(image, width_img=700, height_img=700, questions=5, choices=4, ans=None):
    global score, imgAnswerDisplay
    if ans is None:
        ans = [1, 1, 3, 2, 1]

    #img = cv2.imread(image_path)
    img = cv2.resize(image, (width_img, height_img))

    # preprocessing
    imgContours = img.copy()
    imgBiggestContours = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # Find All contours
    countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, countours, -1, (0, 255, 0), 10)

    # Find Rectangles
    rectCon = utils.rectContour(countours)
    biggestContour = utils.getCornerPoints(rectCon[0])
    Answer = utils.getCornerPoints(rectCon[1])
    ID = utils.getCornerPoints(rectCon[7])


    if Answer.size != 0 and ID.size != 0:
        cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
        biggestContour = utils.reorder(biggestContour)
        Answer = utils.reorder(Answer)
        # Question = utils.reorder(Question)
        pt1 = np.float32(biggestContour)
        pt2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (width_img, height_img))


        # display Answer
        ptG1 = np.float32(Answer)
        ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
        imgAnswerDisplay = cv2.warpPerspective(img, matrixG, (325, 150))


        # apply Threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 190, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = utils.splitBoxes(imgThresh)


        # non zero pixel values of each box
        myPixelVal = np.zeros((questions, choices))
        countC = 0
        countR = 0
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if countC == choices: countR += 1;countC = 0

        myIndex = []
        for x in range(0, questions):
            arr = myPixelVal[x]
            myIndexVal = np.where(arr == np.amax(arr))
            myIndex.append(myIndexVal[0][0])

        # Grading
        grading = []
        for x in range(0, questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)

        # final grade
        score = sum(grading) / questions * 100

    return score,imgAnswerDisplay

"""score,img = grade_test_image('1.jpeg')
try:
    print("SCore: ",score)
    cv2.imshow("Answer",img)
    cv2.waitKey(0)
except cv2.error as e:
    print("opencv error",e)
    """
