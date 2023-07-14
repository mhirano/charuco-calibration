//
// Created by Masahiro Hirano <masahiro.dll@gmail.com>
//

#ifndef ARUCO_H
#define ARUCO_H

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

/** ArUcoマーカーの最大数（表示用） */
//#define ARUCO_MARKER_MAX_NUM					54 // 9*12/2
/** ChArUcoコーナーの最大数（表示用） */
//#define CHARUCO_CORNER_MAX_NUM					88 // (9-1)*(12-1)

struct ArUcoDetector {
	cv::Mat cameraMatrix, distCoeffs;
	int squaresX, squaresY;
	float squareLength, markerLength;
	int dictionaryId;
	bool refindStrategy;
	double gridSize; //mm

	int ARUCO_MARKER_MAX_NUM;
	int CHARUCO_CORNER_MAX_NUM;

	cv::Ptr<cv::aruco::Dictionary> dictionary;
	cv::Ptr<cv::aruco::DetectorParameters> detectorParams;
	cv::Ptr<cv::aruco::CharucoBoard> charucoboard;
	cv::Ptr<cv::aruco::Board> board;
	std::vector<int> markerIds, charucoIds;
	std::vector<std::vector<cv::Point2f>> markerCorners, rejectedMarkers;
	std::vector<cv::Point2f> charucoCorners;
	std::vector<std::vector<cv::Point2f>> markerCornersForShow;
	std::vector<cv::Point2f> charucoCornersForShow;
//	std::vector<cv::Point2f> markerCornersForShow[ARUCO_MARKER_MAX_NUM];
//	cv::Point2f charucoCornersForShow[CHARUCO_CORNER_MAX_NUM];

	ArUcoDetector(
			int _squareX=9, int _squareY=12, double _squareLength=100, double _markerLength=60, int _dictionaryId=10, double _gridSize=100
	) {
		squaresX = _squareX;
		squaresY = _squareY;
		squareLength = _squareLength;
		markerLength = _markerLength;
		dictionaryId = _dictionaryId;
		refindStrategy = true;
		gridSize = _gridSize;
		dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
		detectorParams = cv::aruco::DetectorParameters::create();
		charucoboard = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
		board = charucoboard.staticCast<cv::aruco::Board>();

		ARUCO_MARKER_MAX_NUM = int((_squareX * _squareY) / 2);
		markerCornersForShow.resize(ARUCO_MARKER_MAX_NUM);
		CHARUCO_CORNER_MAX_NUM = (_squareX - 1) * (_squareY - 1);
		charucoCornersForShow.resize(CHARUCO_CORNER_MAX_NUM);

		for (int i = 0; i < ARUCO_MARKER_MAX_NUM; i++) {
			for (int j = 0; j < 4; j++)
				markerCornersForShow[i].push_back(cv::Point2f(0, 0));
		}
		for (int i = 0; i < CHARUCO_CORNER_MAX_NUM; i++) {
			charucoCornersForShow[i] = cv::Point2f(0, 0);
		}
		initClear();
	}
	void initClear() {
		markerIds.clear();
		charucoIds.clear();
		markerCorners.clear();
		rejectedMarkers.clear();
		charucoCorners.clear();
	}
	void detectMarkers(cv::Mat img) {
		detectorParams->cornerRefinementMethod = cv::aruco::CornerRefineMethod::CORNER_REFINE_SUBPIX;

		/** 1. ArUcoマーカー検出 */
		cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, detectorParams, rejectedMarkers);
//		cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, detectorParams, rejectedMarkers, cameraMatrix, distCoeffs); // 古いOpenCVではこちら

		/** 2. ArUcoマーカー洗練 */
		if (refindStrategy) {
			cv::aruco::refineDetectedMarkers(img, board, markerCorners, markerIds, rejectedMarkers, cameraMatrix, distCoeffs, 10.f, 3.f, true, cv::noArray(), detectorParams);
		}
		/** 3. チェスボード検出 */
		int interpolatedCorners = 0;
		if (markerIds.size() > 0)
			interpolatedCorners = cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img, charucoboard, charucoCorners, charucoIds, cameraMatrix, distCoeffs);

		/** 4. 表示用の配列に格納 */
		for (int i = 0; i < (int)markerCorners.size(); i++)
			markerCornersForShow[i] = markerCorners[i];
		for (int i = (int)markerCorners.size(); i < ARUCO_MARKER_MAX_NUM; i++) {
			std::vector<cv::Point2f> markers;
			for (int j = 0; j < 4; j++)
				markers.push_back(cv::Point2f(-1, -1));
			markerCornersForShow[i] = markers;
		}

		for (int i = 0; i < (int)charucoCorners.size(); i++)
			charucoCornersForShow[i] = charucoCorners[i];
		for (int i = (int)charucoCorners.size(); i < CHARUCO_CORNER_MAX_NUM; i++)
			charucoCornersForShow[i] = cv::Point2f(-1, -1);

		/* TODO: Option. 検出結果のサマリーを表示 */
	}

	void drawDetectedMarkers(cv::Mat& img) const {
		std::vector<std::vector<cv::Point2f>> markerCornersShow;
		for (int i = 0; i < ARUCO_MARKER_MAX_NUM; i++) {
			std::vector<cv::Point2f> markers = markerCornersForShow[i];
			if (markers[0].x != -1 && markers[0].y != -1)
				markerCornersShow.push_back(markers);
		}
		if (markerCornersShow.size() > 0)
			cv::aruco::drawDetectedMarkers(img, markerCornersShow, markerIds); // Pass 'markerIds' if ids are needed.
	}

	void drawDetectedCornersCharuco(cv::Mat& img) const {
		std::vector<cv::Point2f> charucoCornersShow;
		for (int i = 0; i < CHARUCO_CORNER_MAX_NUM; i++) {
				cv::Point2f corner = charucoCornersForShow[i];
				if (corner.x != -1 && corner.y != -1)
					charucoCornersShow.push_back(corner);
			}
		if (charucoCornersShow.size() > 0)
			cv::aruco::drawDetectedCornersCharuco(img, charucoCornersShow, charucoIds); // Pass 'charucoIds' if ids are needed.
	}

	void estPatternRT(std::vector<std::vector<cv::Point3d>> markerPoints,
		cv::Mat1d K, cv::Mat1d D, cv::Mat1d &rvec, cv::Mat1d &tvec) {
		std::vector<cv::Point3d> objPoints;
		std::vector<cv::Point2d> imgPoints;
		for (int k = 0; k < markerIds.size(); k++) {
			int id = markerIds[k];

			for (int l = 0; l < markerPoints[id].size(); l++) {
				objPoints.push_back(markerPoints[id][l]);
				imgPoints.push_back(markerCorners[k][l]);
			}
		}
		// マーカーを検出できなかった画像はスキップ
		if (objPoints.size() < 3) return;

		cv::solvePnP(objPoints, imgPoints, K, D, rvec, tvec);
	}

    bool estimatePose(cv::Vec3d& rvec, cv::Vec3d& tvec){
        bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard, cameraMatrix, distCoeffs, rvec, tvec);
        return valid;
    }

};

struct ArUcoCalibrator {
	cv::Mat1d K, D;
	cv::Ptr<cv::aruco::CharucoBoard> charucoboard;
	cv::Ptr<cv::aruco::Board> board;
	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<std::vector<std::vector<cv::Point2f>>> allCorners;
	std::vector<std::vector<int>> allIds;
	std::vector<cv::Mat> allImgs;
	cv::Size imgSize;

	ArUcoCalibrator(int width, int height) {
		K = cv::Mat::eye(3, 3, CV_64FC1);
		D = cv::Mat::zeros(5, 1, CV_64FC1);
//		int IMG_WIDTH = 800;
//		int IMG_HEIGHT = 600;
//		imgSize = cv::Size(IMG_WIDTH, IMG_HEIGHT);
		imgSize = cv::Size(width, height);
	}

	void calibrate(cv::Mat1d &_K, cv::Mat1d &_D, std::vector<cv::Mat> &_Rs, std::vector<cv::Mat> &_Ts,
		std::vector<cv::Mat> &_corners, std::vector<cv::Mat> &_ids) {

		std::vector<std::vector<cv::Point2f>> allCornersConcatenated;
		std::vector<int> allIdsConcatenated;
		std::vector<int> markerCounterPerFrame;
		markerCounterPerFrame.reserve(allCorners.size());
		for (unsigned int i = 0; i < allCorners.size(); i++) {
			markerCounterPerFrame.push_back((int)allCorners[i].size());
			for (unsigned int j = 0; j < allCorners[i].size(); j++) {
				allCornersConcatenated.push_back(allCorners[i][j]);
				allIdsConcatenated.push_back(allIds[i][j]);
			}
		}

		int calibrationFlags = 0;
		double arucoRepErr = cv::aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
			markerCounterPerFrame, board, imgSize, K,
			D, cv::noArray(), cv::noArray(), calibrationFlags);

		int nFrames = (int)allCorners.size();
		std::vector<cv::Mat> allCharucoCorners;
		std::vector<cv::Mat> allCharucoIds;
		std::vector<cv::Mat> filteredImages;
		allCharucoCorners.reserve(nFrames);
		allCharucoIds.reserve(nFrames);

		for (int i = 0; i < nFrames; i++) {
			// interpolate using camera parameters
			cv::Mat currentCharucoCorners, currentCharucoIds;
			cv::aruco::interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charucoboard,
				currentCharucoCorners, currentCharucoIds, K, D);

			allCharucoCorners.push_back(currentCharucoCorners);
			allCharucoIds.push_back(currentCharucoIds);
			filteredImages.push_back(allImgs[i]);
		}

		if (allCharucoCorners.size() < 4) {
			std::cerr << "Not enough corners for calibration" << std::endl;
			return;
		}
		double repError = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard, imgSize, K, D, rvecs, tvecs, calibrationFlags);

		std::cout << "Rep Error: " << repError << std::endl;
		std::cout << "Rep Error Aruco: " << arucoRepErr << std::endl;
		std::cout << K << std::endl;

		_K = K;
		_D = D;
		_Rs = rvecs;
		_Ts = tvecs;
		_corners = allCharucoCorners;
		_ids = allCharucoIds;
	}
};

#endif // ARUCO_H