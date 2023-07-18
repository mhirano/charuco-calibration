#include "CharucoCalibrator.h"

#include "Logger.h"
#include "ArUco.h"

#include <fstream>
#include <regex>

namespace fs = std::filesystem;

/*
 * 保存用構造体
 */
struct MyData{
    int id; cv::Vec3d rvec; cv::Vec3d tvec;
    void write(cv::FileStorage& fs) const //Write serialization for this class
    {
        fs << "{" << "id" << id << "rvec" << rvec << "tvec" << tvec << "}";
    }
};

//These write and read functions must be defined for the serialization in FileStorage to work
static void write(cv::FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}

// This function will print our custom class to the console
static std::ostream& operator<<(std::ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "rvec = " << m.rvec << ", ";
    out << "tvec = " << m.tvec << "}";
    return out;
}

bool CharucoCalibrator::calibrateMono(std::string calibDataPathStr, std::string calibResultPathStr) {
	/*
	 * 0. パス設定
	 */
	std::filesystem::path calibrationDataDirectoryPath(calibDataPathStr);
	std::filesystem::path calibrationResultDirectoryPath(calibResultPathStr);

    /*
     * 1. 画像読み込み
     */
    SPDLOG_INFO("Loading external camera images.");
	std::vector<cv::Mat1b> imgs;
    std::vector<int> idImgs;
    std::vector<std::filesystem::path> paths;
    for (const auto & file : std::filesystem::directory_iterator(calibrationDataDirectoryPath)){
        std::string path = calibrationDataDirectoryPath.string() + "/" + file.path().filename().string(); // Make sure the delimiter is "/"
        paths.emplace_back(std::filesystem::path(path));
    }
    std::sort(paths.begin(), paths.end());
    for (auto i = paths.begin(), e = paths.end(); i != e; ++i) {
        static int idL = 0;
        std::smatch m;
        std::string s = i->string();
        if(i->string().find("external_left") != std::string::npos) { // 画像のプリフィックスを指定．必要に応じて要変更．
            cv::Mat1b img = cv::imread(i->string(), 0);
            imgs.push_back(img);
            if ( std::regex_match(s, m, std::regex(R"(.*/external_left_(.*).png)") )) {
                idImgs.push_back(std::stoi(m[1]));
            }
            SPDLOG_DEBUG("Image: {}", s);
        }
    }
    SPDLOG_INFO("Loaded images.");


    /*
     * 2. キャリブレーション用にArUcoCalibratorを用意
     */
    ArUcoCalibrator arUcoCalibratorLeft(imgs[0].cols, imgs[0].rows);
    ArUcoDetector arUcoDetectorLeft(9,12,100,60,10,100);
    arUcoCalibratorLeft.charucoboard = arUcoDetectorLeft.charucoboard;
    arUcoCalibratorLeft.board = arUcoDetectorLeft.board;

    SPDLOG_INFO("Detecting ArUco.");
    for (int i = 0; i < (int)imgs.size(); i++) {
        SPDLOG_DEBUG("Detecting ArUco No.{}", idImgs[i]);

        cv::Mat markersLeft = imgs[i].clone();
        cv::Mat cornersLeft = imgs[i].clone();

        /* 左画像からArUcoマーカを検知 */
        arUcoDetectorLeft.initClear();
        arUcoDetectorLeft.detectMarkers(imgs[i]);
        arUcoDetectorLeft.drawDetectedMarkers(markersLeft);
        arUcoDetectorLeft.drawDetectedCornersCharuco(cornersLeft);


        bool isUsed;
        if (
                arUcoDetectorLeft.markerCorners.size() > 4 && arUcoDetectorLeft.charucoCorners.size() > 4
                ) {

            arUcoCalibratorLeft.allCorners.push_back(arUcoDetectorLeft.markerCorners);
            arUcoCalibratorLeft.allIds.push_back(arUcoDetectorLeft.markerIds);
            arUcoCalibratorLeft.imgSize = imgs[i].size();
            arUcoCalibratorLeft.allImgs.push_back(imgs[i].clone());
            arUcoCalibratorLeft.K = arUcoDetectorLeft.cameraMatrix.clone();
            arUcoCalibratorLeft.D = arUcoDetectorLeft.distCoeffs.clone();

            isUsed = true;
        } else {
            isUsed = false;
        }

        /* 検知したマーカを保存 */
        std::ostringstream osL, osR, osW;
        osL.clear(); osL.str("");
        osL << calibrationResultDirectoryPath.string() + "/external_left_" << std::setw(4) << std::setfill('0') << idImgs[i] << "_markers.png";
        cv::imwrite(osL.str(), markersLeft.clone());
        osL.clear(); osL.str("");
        osL << calibrationResultDirectoryPath.string() + "/external_left_" << std::setw(4) << std::setfill('0') << idImgs[i] << "_corners.png";
        cv::imwrite(osL.str(), cornersLeft.clone());

        /* 検知結果のサマリーを表示 */
        SPDLOG_DEBUG("Image (left) {}: Detected markers {}, corners {}, isUsed {}",
                     i,
                     arUcoDetectorLeft.markerCorners.size(),
                     arUcoDetectorLeft.charucoCorners.size(),
                     isUsed);

		{
			cv::imwrite("markers_"+std::to_string(i)+".png", markersLeft);
			cv::imwrite("corners_"+std::to_string(i)+".png", cornersLeft);
		}
    }

    /*
     * 3. 各カメラの単眼キャリブレーション
     */
    cv::Mat1d K,D;
    std::vector<cv::Mat> rots;
    std::vector<cv::Mat> trans;
    std::vector<cv::Mat> corners;
    std::vector<cv::Mat> ids;
    SPDLOG_INFO("Calibrating each camera. Hold on!");

    arUcoCalibratorLeft.calibrate(K, D, rots, trans, corners, ids);
    SPDLOG_INFO("Calibrated each camera.");

	/*
	 * キャリブレーション結果の保存
 */
    SPDLOG_INFO("Saving calibration results.");
    std::string externalCameraCalibrationResult = "MonocularCameraCalibrationResult.xml";
	std::ostringstream os;
	os.clear(); os.str("");
	os << calibrationResultDirectoryPath.string() + "/" + externalCameraCalibrationResult;
	cv::FileStorage fs(os.str(), cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		SPDLOG_ERROR("Monocular camera calibration result file {} can not be opened.");
		return false;
	}

	// 単眼
	fs << "K" << K;
	fs << "D" << D;

	fs.release();

	SPDLOG_DEBUG("Monocular camera calibration completed.");

    return true;

}


bool CharucoCalibrator::calibrateStereo(std::string calibDataPathStr, std::string calibResultPathStr) {

    /*
     * 0. パス設定
     */
    std::filesystem::path calibrationDataDirectoryPath(calibDataPathStr);
    std::filesystem::path calibrationResultDirectoryPath(calibResultPathStr);

    /*
     * 1. 画像読み込み
     */
	SPDLOG_INFO("Loading external camera images.");
	std::vector<cv::Mat1b> imgsL, imgsR;
	std::vector<int> idImgsL, idImgsR;
	std::vector<std::filesystem::path> paths;
	for (const auto & file : std::filesystem::directory_iterator(calibrationDataDirectoryPath)){
		std::string path = calibrationDataDirectoryPath.string() + "/" + file.path().filename().string(); // Make sure the delimiter is "/"
		paths.emplace_back(std::filesystem::path(path));
	}
	std::sort(paths.begin(), paths.end());
	for (auto i = paths.begin(), e = paths.end(); i != e; ++i) {
		static int idL = 0, idR = 0;
		std::smatch m;
		std::string s = i->string();
		if(i->string().find("external_left") != std::string::npos) {
			cv::Mat1b imgL = cv::imread(i->string(), 0);
			imgsL.push_back(imgL);
			if ( std::regex_match(s, m, std::regex(R"(.*/external_left_(.*).png)") )) {
				idImgsL.push_back(std::stoi(m[1]));
			}
			SPDLOG_DEBUG("External camera image left: {}", s);
		}
		if(i->string().find("external_right") != std::string::npos){
			cv::Mat1b imgR = cv::imread(i->string(), 0);
			imgsR.push_back(imgR);
			if ( std::regex_match(s, m, std::regex(R"(.*/external_right_(.*).png)") )) {
				idImgsR.push_back(std::stoi(m[1]));
			}
			SPDLOG_DEBUG("External camera image right: {}", s);
		}
	}
	assert((imgsL.size() == imgsR.size()) && "Left, right and wide images should have same number of images");
	if(imgsL.empty() || imgsR.empty()) return false;
	SPDLOG_INFO("Loaded external camera images.");

	/*
	 * 2. キャリブレーション用にArUcoCalibratorを用意
	 */
	ArUcoCalibrator arUcoCalibratorLeft(imgsL[0].cols, imgsL[0].rows);
	ArUcoDetector arUcoDetectorLeft(9,12,100,60,10,100);
	arUcoCalibratorLeft.charucoboard = arUcoDetectorLeft.charucoboard;
	arUcoCalibratorLeft.board = arUcoDetectorLeft.board;

	ArUcoCalibrator arUcoCalibratorRight(imgsR[0].cols, imgsR[0].rows);
	ArUcoDetector arUcoDetectorRight(9,12,100,60,10,100);
	arUcoCalibratorRight.charucoboard = arUcoDetectorRight.charucoboard;
	arUcoCalibratorRight.board = arUcoDetectorRight.board;

    SPDLOG_INFO("Detecting ArUco.");
	for (int i = 0; i < (int)imgsL.size(); i++) {
		SPDLOG_DEBUG("Detecting ArUco No.{}", idImgsL[i]);

		cv::Mat markersLeft = imgsL[i].clone();
		cv::Mat cornersLeft = imgsL[i].clone();
		cv::Mat markersRight = imgsR[i].clone();
		cv::Mat cornersRight = imgsR[i].clone();

		/* 左画像からArUcoマーカを検知 */
		arUcoDetectorLeft.initClear();
		arUcoDetectorLeft.detectMarkers(imgsL[i]);
		arUcoDetectorLeft.drawDetectedMarkers(markersLeft);
		arUcoDetectorLeft.drawDetectedCornersCharuco(cornersLeft);

		/* 右画像からArUcoマーカを検知 */
		arUcoDetectorRight.initClear();
		arUcoDetectorRight.detectMarkers(imgsR[i]);
		arUcoDetectorRight.drawDetectedMarkers(markersRight);
		arUcoDetectorRight.drawDetectedCornersCharuco(cornersRight);

		bool isUsed;
		if (
			arUcoDetectorLeft.markerCorners.size() > 4 && arUcoDetectorLeft.charucoCorners.size() > 4
			&& arUcoDetectorRight.markerCorners.size() > 4 && arUcoDetectorRight.charucoCorners.size() > 4
		) {

			arUcoCalibratorLeft.allCorners.push_back(arUcoDetectorLeft.markerCorners);
			arUcoCalibratorLeft.allIds.push_back(arUcoDetectorLeft.markerIds);
			arUcoCalibratorLeft.imgSize = imgsL[i].size();
			arUcoCalibratorLeft.allImgs.push_back(imgsL[i].clone());
			arUcoCalibratorLeft.K = arUcoDetectorLeft.cameraMatrix.clone();
			arUcoCalibratorLeft.D = arUcoDetectorLeft.distCoeffs.clone();

			arUcoCalibratorRight.allCorners.push_back(arUcoDetectorRight.markerCorners);
			arUcoCalibratorRight.allIds.push_back(arUcoDetectorRight.markerIds);
			arUcoCalibratorRight.imgSize = imgsR[i].size();
			arUcoCalibratorRight.allImgs.push_back(imgsR[i].clone());
			arUcoCalibratorRight.K = arUcoDetectorRight.cameraMatrix.clone();
			arUcoCalibratorRight.D = arUcoDetectorRight.distCoeffs.clone();

			isUsed = true;
		} else {
			isUsed = false;
		}

		/* 検知したマーカを保存 */
		std::ostringstream osL, osR, osW;
		osL.clear(); osL.str("");
		osL << calibrationResultDirectoryPath.string() + "/external_left_" << std::setw(4) << std::setfill('0') << idImgsL[i] << "_markers.png";
		cv::imwrite(osL.str(), markersLeft.clone());
		osL.clear(); osL.str("");
		osL << calibrationResultDirectoryPath.string() + "/external_left_" << std::setw(4) << std::setfill('0') << idImgsL[i] << "_corners.png";
		cv::imwrite(osL.str(), cornersLeft.clone());
		osR.clear(); osR.str("");
		osR << calibrationResultDirectoryPath.string() + "/external_right_" << std::setw(4) << std::setfill('0') << idImgsR[i] << "_markers.png";
		cv::imwrite(osR.str(), markersRight.clone());
		osR.clear(); osR.str("");
		osR << calibrationResultDirectoryPath.string() + "/external_right_" << std::setw(4) << std::setfill('0') << idImgsR[i] << "_corners.png";
		cv::imwrite(osR.str(), cornersRight.clone());

		/* 検知結果のサマリーを表示 */
		SPDLOG_DEBUG("Image (left) {}: Detected markers {}, corners {}, isUsed {}",
					 i,
					 arUcoDetectorLeft.markerCorners.size(),
					 arUcoDetectorLeft.charucoCorners.size(),
					 isUsed);
		SPDLOG_DEBUG("Image (right) {}: Detected markers {}, corners {}, isUsed {}",
					 i,
					 arUcoDetectorRight.markerCorners.size(),
					 arUcoDetectorRight.charucoCorners.size(),
					 isUsed);

        {
            cv::imwrite("markersLeft_"+std::to_string(i)+".png", markersLeft);
            cv::imwrite("cornersLeft_"+std::to_string(i)+".png", cornersLeft);
            cv::imwrite("markersRight_"+std::to_string(i)+".png", markersRight);
            cv::imwrite("cornersRight_"+std::to_string(i)+".png", cornersRight);
        }
	}

	/*
	 * 3. 各カメラの単眼キャリブレーション
	 */
	cv::Mat1d KL, KR, KW, DL, DR, DW;
	std::vector<cv::Mat> rotsL, rotsR, rotsW;
	std::vector<cv::Mat> transL, transR, transW;
	std::vector<cv::Mat> cornersL, cornersR, cornersW;
	std::vector<cv::Mat> idsL, idsR, idsW;
	SPDLOG_INFO("Calibrating each camera. Hold on!");

	arUcoCalibratorLeft.calibrate(KL, DL, rotsL, transL, cornersL, idsL);
	arUcoCalibratorRight.calibrate(KR, DR, rotsR, transR, cornersR, idsR);
	SPDLOG_INFO("Calibrated each camera.");

	/*
	 * 4. ステレオキャリブレーション
	 *     単眼キャリブレーションの結果を構造体に格納
	 *     左カメラ、右カメラのステレオキャリブレーション
	 */
    SPDLOG_INFO("Calibrating stereo camera.");
	ArUcoDetector arUcoDetector(9,12,100,60,10,100);
	int squaresX = arUcoDetector.squaresX;
	int squaresY = arUcoDetector.squaresY;
	double gridSize = arUcoDetector.gridSize; // [mm]
	assert("Image sizes of left and right cameras must be same." && (imgsL[0].cols == imgsR[0].cols) && (imgsL[0].rows == imgsR[0].rows));
	cv::Size imgSize(imgsL[0].cols, imgsL[0].rows);

	std::vector<std::vector<cv::Point2f>> CPsL, CPsR;
	std::vector<std::vector<cv::Point3f>> worldPoints;

	/* 左右両方で見えているコーナーを利用 */
	for (int i = 0; i < cornersL.size(); i++) {
		std::vector<cv::Point2f> pointsL, pointsR;
		std::vector<cv::Point3f> worldPointsBuf;

		int p_num = (squaresX - 1) * (squaresY - 1);
		for (int j = 0; j < p_num; j++) {
			int vis_count = 0;
			cv::Point2f pl, pr;
			for (int k = 0; k < idsL[i].rows; k++) {
				int idL = idsL[i].at<int>(k);
				if (idL == j) {
					pl.x = cornersL[i].at<float>(k, 0);
					pl.y = cornersL[i].at<float>(k, 1);
					vis_count++;
				}
			}
			for (int k = 0; k < idsR[i].rows; k++) {
				int idR = idsR[i].at<int>(k);
				if (idR == j) {
					pr.x = cornersR[i].at<float>(k, 0);
					pr.y = cornersR[i].at<float>(k, 1);
					vis_count++;
				}
			}
			/* 両方の画像で見えている点を抽出 */
			if (vis_count == 2) {
				pointsL.push_back(pl);
				pointsR.push_back(pr);
				worldPointsBuf.emplace_back(cv::Point3f(static_cast<float>(j % (squaresX - 1) * gridSize), static_cast<float>(j / (squaresX - 1) * gridSize), 0.0));
			}
		}

		// 十分検出できなかったものはreject
		if(pointsL.size() >= 4 && pointsR.size() >= 4) {
			CPsL.push_back(pointsL);
			CPsR.push_back(pointsR);
			worldPoints.push_back(worldPointsBuf);
		}
	}

	/* ステレオ校正 */
	cv::Mat1d E, F, R, T;
	double error = cv::stereoCalibrate(worldPoints, CPsL, CPsR,
		KL, DL, KR, DR, imgSize,
		R, T, E, F,
		cv::CALIB_FIX_INTRINSIC, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-5));
    SPDLOG_INFO("Calibrated stereo camera.");

    std::cout << "R: " << R << std::endl;
    std::cout << "T: " << T << std::endl;

	/*
	 * キャリブレーション結果の保存
	 */
    SPDLOG_INFO("Saving calibration results.");
    std::string externalCameraCalibrationResult = "StereoCameraCalibrationResult.xml";
	std::ostringstream os;
	os.clear(); os.str("");
	os << calibrationResultDirectoryPath.string() + "/" + externalCameraCalibrationResult;
	cv::FileStorage fs(os.str(), cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		SPDLOG_ERROR("Stereo camera calibration result file {} can not be opened.");
		return false;
	}

	// 単眼
	fs << "Left_K" << KL;
	fs << "Left_D" << DL;
	fs << "Right_K" << KR;
	fs << "Right_D" << DR;

	// ステレオ
	fs << "Stereo_Error" << error;
	fs << "Left_Right_R" << R;
	fs << "Left_Right_T" << T;
	fs << "Left_Right_E" << E;
	fs << "Left_Right_F" << F;

	fs.release();

	SPDLOG_DEBUG("Stereo camera calibration completed.");

	return true;
}

bool CharucoCalibrator::estimateBoardPose(std::string calibDataPathStr, std::string calibResultPathStr){
	/**
	 * 1. ボード画像たちの読み込み
	 * 2. 内部パラメータの読み込み
	 * 3. ボード画像たちからコーナー点検出＆Refine
	 * 4. ボードの姿勢推定 (cv::aruco::estimatePoseCharucoBoard)
	 * 5. 結果の保存
	 */
	/*
	 * 0. パス設定
	 */
	std::filesystem::path calibrationDataDirectoryPath(calibDataPathStr);
	std::filesystem::path calibrationResultDirectoryPath(calibResultPathStr);

	/*
     * 1. 画像読み込み
     */
	SPDLOG_INFO("Loading external camera images.");
	std::vector<cv::Mat1b> imgs;
	std::vector<int> idImgs;
	std::vector<std::filesystem::path> paths;
	for (const auto & file : std::filesystem::directory_iterator(calibrationDataDirectoryPath)){
		std::string path = calibrationDataDirectoryPath.string() + "/" + file.path().filename().string(); // Make sure the delimiter is "/"
		paths.emplace_back(std::filesystem::path(path));
	}
	std::sort(paths.begin(), paths.end());
	for (auto i = paths.begin(), e = paths.end(); i != e; ++i) {
		static int idL = 0;
		std::smatch m;
		std::string s = i->string();
		if(i->string().find("external_left") != std::string::npos) { // 画像のプリフィックスを指定．必要に応じて要変更．
			cv::Mat1b img = cv::imread(i->string(), 0);
			imgs.push_back(img);
			if ( std::regex_match(s, m, std::regex(R"(.*/external_left_(.*).png)") )) {
				idImgs.push_back(std::stoi(m[1]));
			}
			SPDLOG_DEBUG("Image: {}", s);
		}
	}
	SPDLOG_INFO("Loaded images.");

    /*
     * 推定結果保存用ストリームの準備
     */
    std::string externalCameraCalibrationResult = "poses.xml";
    std::ostringstream os;
    os.clear(); os.str("");
    os << calibrationResultDirectoryPath.string() + "/" + externalCameraCalibrationResult;
    cv::FileStorage fs(os.str(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        SPDLOG_ERROR("Pose estimation result file {} can not be opened.");
        return false;
    }


    /*
     * 2.
     */

    ArUcoCalibrator arUcoCalibratorLeft(imgs[0].cols, imgs[0].rows);
	ArUcoDetector arUcoDetectorLeft(9,12,100,60,10,100);
	arUcoCalibratorLeft.charucoboard = arUcoDetectorLeft.charucoboard;
	arUcoCalibratorLeft.board = arUcoDetectorLeft.board;

	SPDLOG_INFO("Detecting ArUco.");
	for (int i = 0; i < (int)imgs.size(); i++) {
		SPDLOG_DEBUG("Detecting ArUco No.{}", idImgs[i]);

		cv::Mat markerAndCornerLeft = imgs[i].clone();
//        cv::Mat axisImg = imgs[i].clone();
//        cv::cvtColor(axisImg, axisImg, cv::COLOR_GRAY2BGR);

		/* 左画像からArUcoマーカを検知 */
		arUcoDetectorLeft.initClear();
        // avs
		float fx = 2.597878086937441e+03, fy = 2.597608077430456e+03, cx = 2.151484663510314e+02, cy = 2.165581942395036e+02;
        float k1 = -0.803435516039213, k2 = 86.400017750811530, k3 = -4.648338417345626e+03; // 半径, radial
		float p1 = 0.001105847484596, p2 = 0.004106860666786; // 円周, tangential
        // falcon
//        float fx = 1.8793735543302303e+03, fy = 1.8044510334566894e+03, cx = 1.4324188639493905e+03, cy = 9.1631672262384984e+02;
//        float k1 = 1.2565438183754463e-01, k2 = -2.2261018793922411e+00, k3 = 1.5935431887100768e+00; // 半径, radial
//        float p1 =  -6.1700664086824696e-02, p2 = -1.9218289296846904e-02; // 円周, tangential

		arUcoDetectorLeft.cameraMatrix = (cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1); // [fx, 0, cx; 0, fy, cy; 0 0 1]
		arUcoDetectorLeft.distCoeffs = (cv::Mat_<float>(1,5) << k1, k2, p1, p2, k3); // [fx, 0, cx; 0, fy, cy; 0 0 1]
		arUcoDetectorLeft.detectMarkers(imgs[i]);


        cv::Vec3d rvec, tvec;
        {
            arUcoDetectorLeft.estPatternRT(rvec, tvec);
            std::cout << "tvec(pnp):" << tvec<< std::endl;
            std::cout << "rvec(pnp):" << rvec<< std::endl;
			cv::Mat axisImg = imgs[i].clone();
			cv::cvtColor(axisImg, axisImg, cv::COLOR_GRAY2BGR);
			cv::aruco::drawAxis(axisImg, arUcoDetectorLeft.cameraMatrix, arUcoDetectorLeft.distCoeffs, rvec, tvec, 100.0f);
			cv::imshow("axis(pnp)", axisImg);
		}
		if( arUcoDetectorLeft.estimatePose(rvec, tvec) ){
			cv::Mat axisImg = imgs[i].clone();
			cv::cvtColor(axisImg, axisImg, cv::COLOR_GRAY2BGR);
			// draw axis
            cv::aruco::drawDetectedCornersCharuco( markerAndCornerLeft, arUcoDetectorLeft.charucoCorners, arUcoDetectorLeft.charucoIds, cv::Scalar(255,0,0) );
            cv::aruco::drawDetectedMarkers( markerAndCornerLeft, arUcoDetectorLeft.markerCorners, arUcoDetectorLeft.markerIds );
            cv::aruco::drawAxis(axisImg, arUcoDetectorLeft.cameraMatrix, arUcoDetectorLeft.distCoeffs, rvec, tvec, 100.0f);

            std::cout << "tvec:" << tvec<< std::endl;
            std::cout << "rvec:" << rvec<< std::endl;
            cv::Mat rot;
            cv::Rodrigues(rvec, rot);
            std::cout << "rmat:" << rot << std::endl;
			std::cout << "i: " << i << std::endl;

            cv::imshow("marker_and_corner", markerAndCornerLeft);
            cv::imshow("axis(ocv)", axisImg);
//            cv::waitKey(0);

            MyData m{idImgs[i], rvec, tvec};

            // 単眼
            fs << "data" << m;
        }

	}

    fs.release();

    SPDLOG_INFO("Pose estimation completed.");

    return true;
}