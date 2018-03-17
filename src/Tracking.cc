/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, const bool bReuse):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(bReuse), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef); //畸变系数

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    is_preloaded = bReuse;
	if (is_preloaded)
    {
		mState = LOST;
    }
    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR) //mpIniORBextractor相比mpORBextractorLeft提取的特征点多一倍
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    cout << "- Reuse Map ?: " << is_preloaded << endl;
    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(mDepthMapFactor==0)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mDepthMapFactor!=1 || imDepth.type()!=CV_32F);
    imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

/**
 * @brief
 * @param 左目RGB或RGBA图像与时间戳
 * 1. 转换图像
 * 2. 构造Frame
 * 3. 进入tracking线程
 * 4. 位姿计算
 * @return 世界坐标系到相机坐标系的变换矩阵
 */
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    // 【步骤1】: 转换图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // 【步骤2】: 构造Frame
    // 没有初始化与初始化完成构造的frame不一样
    // mpIniORBextractor相比mpORBextractorLeft提取的特征点多一倍
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // 【步骤3】: 进入tracking线程
    Track();

    // 【步骤4】: 位姿计算，位姿返回给main函数
    /* Do Pose calculation */
    if(mState==OK && !mCurrentFrame.mTcw.empty() && mCurrentFrame.mpReferenceKF)    
    {

        vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId); //使用Id号进行排序

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        // 转换全部的关键帧为了第一个关键者是起始帧
        // 有了闭环之后，第一帧可能不再是起始帧了

        cv::Mat Two = vpKFs[0]->GetPoseInverse(); //Id=0表示第一帧

        ORB_SLAM2::KeyFrame* pKF = mpReferenceKF;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);
        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }
        Trw = Trw*pKF->GetPose()*Two; 
        cv::Mat Tcr = mlRelativeFramePoses.back();
        cv::Mat Tcw = Tcr*Trw;
        return Tcw.clone();
    }
    else 
        return mCurrentFrame.mTcw.clone();
}

/**
 * @brief tracking线程
 * 1. 单目初始化(对极几何)
 * 2. 跟踪
 *    2.1 跟踪上一帧或者参考帧
 *    2.2 跟踪失败，触发重定位
 * 3.
 */
void Tracking::Track()
{
    // track包含两部分: 估计运动、跟踪局部地图

    // mState为tracking的状态机
    // SYSTEM_NOT_READY，NO_IMAGES_YET，NOT_INITIALIZED，OK，LOST
    // 如果图像复位过、或者第一次运行，则为NO_IMAGES_YET状态
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED; //NO_IMAGES_YET状态不会再次进入了
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate); //给map加锁

    /*【步骤1】: 初始化*/
    if(mState==NOT_INITIALIZED)
    {
        if (is_preloaded)
        {
            mState = LOST;
            return;
        }
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else /*【步骤1.1】: 单目初始化*/
            MonocularInitialization(); //初始化成功后将mState=OK

        mpFrameDrawer->Update(this); //Update info from the last processed frame.

        if(mState!=OK)
            return;
    }
    else /*【步骤2】: 跟踪*/
    {
        // System is initialized. Track Frame.

        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 在viewer中有个开关menulocalizationMode,有它控制是否AactivateLocalizationMode，并最终控制mbOnlyTracking
        // mbOnlyTracking等于false表示同时建图与定位模式，mbOnlyTracking等于true表示用户手动选择定位模式
/*slam模式*/
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
    /*【步骤2.1】: 跟踪上一帧或者参考帧*/

            if(mState==OK)// 初始化成功、或者跟踪成功
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 检查并更新上一帧的MapPoints
                // 对于双目和RGBD，updateLastFrame函数会为上一帧添加MapPoints
                // 上一帧的MapPoints主要是为了SearchByProjection(TrackWithMotionModel时才会用到)时加速匹配
                CheckReplacedInLastFrame(); /** @todo */

                // 运动模型是空或刚完成重定位
                // mnLastRelocFrameId为上一次重定位的那一帧
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // 将<上一帧的位姿>作为<当前帧初始位姿>
                    // 通过<BoW>的方式在<参考关键帧>中找到当前帧特征点的匹配点
                    // 优化每个特征点对应的3D点重投影误差即可得到位姿
                    bOK = TrackReferenceKeyFrame();
                }
                else // mVelocity不为空，就优先选择TrackWithMotionModel
                {
                    // 根据<恒速模型>设定当前帧的初始位姿
                    // 通过<SearchByProjection函数>计算参考关键帧与当前帧匹配点
                    // 优化每个特征点对应的3D点重投影误差即可得到位姿
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else /*【步骤2.2】: 跟踪失败，触发重定位*/
            {
                //BoW搜索，PnP求解位姿
                bOK = Relocalization();
            }
        }
/*定位模式*/
        else
        {
            // Only Tracking: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                // mbVO为false表示上一帧追踪到足够多的MapPoints，mbVO为true表示仅运行VO(视觉里程计)
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    // 同时计算relocalization与visual odometry，但更相信relocalization

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc) //只要<重定位>成功整个跟踪过程正常进行(定位与跟踪，更相信定位)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // 将最新的关键帧作为reference frame
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
    /*【步骤2.3】: 在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿*/
        if(!mbOnlyTracking)
        {
    /*【步骤2.3.1】: 同时定位与建图模式，只需要判断是否<跟踪或者重定位>成功*/
            if(bOK)
                bOK = TrackLocalMap(); //局部地图跟踪
        }
        else
        {
            // mbVO true means that there are few(很少) matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
    /*【步骤2.3.2】: <跟踪或者重定位>成功、并且不是VO模式*/
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        // 追踪成功， 检查是否需要插入关键帧
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
    /*【步骤2.4】: 更新恒速运动模型中(TrackWithMotionModel函数)的Velocity*/
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc; //对，相当于*LastTcw.t()
            }
            else
                mVelocity = cv::Mat(); //空，NULL

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean temporal point matches
    /*【步骤2.5】: 清除当前帧观测不到的那些map中的3D点*/
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    // 主要用于排除UpdateLastFrame函数中<为了跟踪增加>的MapPoints
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
    /*【步骤2.6]: 清除临时的MapPoints,这里的MapPoints在TrackWithMotionModel里生成的(仅双目和RGBD)*/
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
    /*【步骤2.7]: 检测并插入关键帧*/
            if(NeedNewKeyFrame()) /** @todo*/
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
    /*【步骤2.8]: 如果在初始化不久状态就丢失了，则复位系统*/
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF; //【迭代】

        mLastFrame = Frame(mCurrentFrame); //【迭代】
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty() && mCurrentFrame.mpReferenceKF)
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}
#if 0
cv::Mat Tracking::getTransformData()
{   
        return mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
}
#endif


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

/**
 * @brief 单目初始化
 */
void Tracking::MonocularInitialization()
{

    if(!mpInitializer) //对象指针是否为空?
    {
        // Set Reference Frame
    /*【步骤1】: 设置参考帧*/
        if(mCurrentFrame.mvKeys.size()>100) //关键点数目大于100
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
		else
    		cout << __FUNCTION__ << "The Key Frame-s points are less: " << mCurrentFrame.mvKeys.size() << endl;
    }
    else
    {
        // Try to initialize
    /*【步骤2】: 尝试初始化*/
        if((int)mCurrentFrame.mvKeys.size()<=100) //特征点数目少于100, 重新初始化
        {
			
    		//cout << __FUNCTION__ << "old.The Key Frame-s points are less: " << mCurrentFrame.mvKeys.size() << endl;
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
    /*【步骤2.1】: 特征点两两匹配，寻找前后两帧图像之间的匹配*/
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100); /** @todo */

        // Check if there are enough correspondences
        if(nmatches<100) //匹配数目少于100, 重新初始化
        {
    		cout << __FUNCTION__ << "ORB extraction : No enough correspondesnces(<100) " << nmatches << endl;
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

    /*【步骤2.2】: 通过Initialize函数计算基础矩阵与单应矩阵，得到<Rcw、tcw>以及<MapPoints坐标>*/
        // mvIniP3D是worldpose，也即是相对mInitialFrame坐标系的坐标，因为mInitialFrame的位姿设置为单位矩阵
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) /** @todo */
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i]) //三角化标志
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
    /*【步骤2.3】: 设置mInitialFrame、mCurrentFrame的位姿*/
            // 将mInitialFrame对应的坐标系作为世界坐标系
            // mInitialFrame位姿为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F)); //单位矩阵
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

    /*【步骤2.4】: 初始化完成后，创建地图*/
            CreateInitialMapMonocular();
        }
    }
}

/**
 * @brief Tracking::CreateInitialMapMonocular
 * 1、创建关键帧
 * 2、向地图中插入关键帧
 * 3、创建MapPoints
 *  3.1、向<关键帧>中添加<MapPoint>，与3.2是互相的关系
 *  3.2、向该<MapPoint>添加能观测到它的<关键帧>
 *  3.3、计算MapPoint的描述子
 *  3.4、填充当前帧
 * 4、向地图中插入MapPoint
 * .
 * .
 * .
 */
void Tracking::CreateInitialMapMonocular()
{

    cout << __FUNCTION__ << ": Starting Initial Map Creation..." << endl;
    // Create KeyFrames
    /*【步骤1】: 创建关键帧*/
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB); //将<初始帧>生成<初始关键帧>
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB); //将<当前帧>生成<当前关键帧>


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    /*【步骤2】: 向地图中插入关键帧*/
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    /*【步骤3】: 创建MapPoints*/
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0) //无效的匹配点跳过
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

    /*【步骤3.1】: 向<关键帧>中添加<MapPoint>，与3.2是互相的关系*/
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    /*【步骤3.2】: 向该<MapPoint>添加能观测到它的<关键帧>*/
        pMP->AddObservation(pKFini, i); //关键帧+该MapPoint在此关键帧index
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

    /*【步骤3.3】: 计算MapPoint的描述子*/
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
    /*【步骤3.4】: 填充当前帧*/
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP; //将匹配上的MapPoints放到mCurrentFrame中
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
    /*【步骤4】: 向地图中插入MapPoint*/
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    /*【步骤5】: 更新连接*/
    pKFini->UpdateConnections(); /** @todo */
    pKFcur->UpdateConnections();

    // 新地图(初始化)被创建具有多少个MapPoints
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // Bundle Adjustment
    /*【步骤6】: 全局BA优化*/
    Optimizer::GlobalBundleAdjustemnt(mpMap,20); /** @todo */

    // Set median depth to 1
    /*【步骤7】: 计算深度的<中值>*/
    float medianDepth = pKFini->ComputeSceneMedianDepth(2); /** @todo */
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting... map points(100):" 
		<< pKFcur->TrackedMapPoints(1) 
		<< " medianDepth = " << medianDepth << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    /*【步骤8】: 使用<步骤7计算的深度中值>调整<当前关键帧>位姿中pose尺度*/
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth; //只对位姿中的pose进行尺度调整
    pKFcur->SetPose(Tc2w);

    // Scale points
    /*【步骤9】: 使用<步骤7计算的深度中值>调整<初始关键帧>中MapPoints的尺度*/
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth); /** @todo GetWorldPos函数需优化*/
        }
    }

    /*【步骤10】: 向LocalMapping中添加关键帧*/
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    /*【步骤11】: 步骤8中改变了位姿的尺度，故需要再次setpose(MonocularInitialization函数步骤2.3执行过一次)*/
    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId; // 【迭代】id号
    mpLastKeyFrame = pKFcur; // 【迭代】当前关键帧-->>上一关键帧

    /*【步骤11】: 局部地图，压栈关键帧与MapPoints*/
    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur; //【迭代】当前关键帧-->>参考关键帧
    mCurrentFrame.mpReferenceKF = pKFcur; //【迭代】当前关键帧-->>当前帧的<参考关键帧>

    mLastFrame = Frame(mCurrentFrame); // 【迭代】当前普通帧-->>上一普通关键帧

    // 地图中设置参考地图点
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    /*【步骤12】: 绘图更新*/
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    /*【步骤13】: 将pKFini作为关键帧起始关键帧*/
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    /*【步骤14】: 状态->OK*/
    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced(); //替换?
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true); //0.7: the ratio of the best and the second score，越小匹配越苛刻
    vector<MapPoint*> vpMapPointMatches;

    /*【步骤1】: 通过词袋(BoW)搜索得到匹配点，得到与mCurrentFrame中的keyPoints一样排序的MapPoints*/
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches); /** @done */

    // 匹配点数目少于15,不做任何事
    if(nmatches<15)
        return false;

    /*【步骤1.1】: 设置mCurrentFrame观测到的(参考帧与当前帧匹配上的)MapPoints*/
    // mvpMapPoints: MapPoints associated to keypoints, NULL pointer if no association.
    // vector中的MapPoint为NULL表示此keyPoint没有对应的MapPoint
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    /*【步骤1.2】: 将上一帧的位姿作为当前帧的位姿*/
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    /*【步骤2】: 位姿BA优化*/
    Optimizer::PoseOptimization(&mCurrentFrame); /** @todo */

    // Discard outliers
    /*【步骤3】: 抛弃误匹配的3D点(MapPoints)*/
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) //Observations函数获取能观测到该MapPoint的关键帧数目
                nmatchesMap++; //匹配成功次数加1
        }
    }

    return nmatchesMap>=10;
}

/**
 * @brief 只用于双目或者RGBD
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points
    /*【步骤1】: 根据上一帧对应的参考关键帧更新上一帧的位姿*/
    UpdateLastFrame(); //只对双目和rgbd起作用

    /*【步骤2】: 设置mCurrentFrame的位姿*/
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    // 清空mCurrentFrame中的MapPoints
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;

    /*【步骤3】: 通过SearchByProjection函数计算匹配点*/
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    // 如果匹配点数目较少，则使用更宽的的窗口搜索
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    /*【步骤4】: 使用匹配点BA优化位姿*/
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    /*【步骤5】: 抛弃误匹配的3D点(MapPoints)*/
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--; //检测到误匹配点，匹配数也要减去
            }/*【步骤6】: <观测到该MapPoint>的<关键帧数目>是否大于0*/
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    /*【步骤7】: 定位模式下，跟踪成功还是只有VO?*/
    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10; //仅有VO
        return nmatches>20; //跟踪成功
    }

    return nmatchesMap>=10;
}

/**
 * @brief Tracking::TrackLocalMap
 * @return
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    /*【步骤1]: 更新局部地图,包括局部关键帧mvpLocalKeyFrames与局部地图点mvpLocalMapPoints*/
    UpdateLocalMap();

    /*【步骤2]: 搜索局部地图点，将符合条件的添加到mCurrentFrame中*/
    SearchLocalPoints();

    // Optimize Pose
    /*【步骤3]: 优化位姿*/
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); /** @todo*/
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // 刚重定位成功要求更严格，why?
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    /*【步骤4]: 匹配成功数目是否大于30 */
    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/**
 * @brief Tracking::NeedNewKeyFrame
 * @return
 */
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    /* 【步骤1]: */
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap(); /** @todo */

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    /* 【步骤2]: 满足一下两个条件:(see paper V-E)*/
    // 1、如果没有足够多的帧(mMaxFrames)从上一次重定位中传递下来，则不要插入关键帧
    // 2、地图中的关键数量大于mMaxFrames
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // 参考关键帧中被跟踪的MapPoints数目
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs); //参考关键帧中被跟踪的MapPoints数目

    // Local Mapping accept keyframes?
    // LocalMapping 是否空闲
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
    // "total matches = matches to map + visual odometry matches"
    // Visual odometry matches will become MapPoints if we insert a keyframe.
    // This ratio measures how many MapPoints we could create if we insert a keyframe.
    int nMap = 0; //matches to map
    int nTotal= 0; //total matches= matches to map + visual odometry matches
    // 立体视觉
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                nTotal++;
                if(mCurrentFrame.mvpMapPoints[i])
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) //计算matches to map
                        nMap++;
            }
        }
    } //单目
    else
    {
        // There are no visual odometry matches in the monocular case
        nMap=1;
        nTotal=1;
    }

    const float ratioMap = (float)nMap/fmax(1.0f,nTotal);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    // Current frame tracks less than 90% points than Kref.
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    float thMapRatio = 0.35f;
    if(mnMatchesInliers > 300) //当前帧匹配成功的点，在TrackLocalMap函数中计算
        thMapRatio = 0.20f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 是否有MaxFrames个帧自从上一次关键帧开始传递下来
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // 是否有MinFrames个帧自从上一次关键帧开始传递下来、以及局部建图是否空闲
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    // 跟踪很弱、很不稳定
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || ratioMap<0.3f) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 满足两个条件:
    // 1、当前帧匹配到的Map Point的个数不能超过参考关键帧（与当前帧拥有最多map point的关键帧）对应Map Point的90%
    // 2、当前的帧匹配到的Map Point大于15
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio || ratioMap<thMapRatio) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
    /*【步骤3]: 如果mapping接受keyframes，则插入keyframe，否则发送一个中断BA的信号*/
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA(); //中断BA，插入关键帧
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    /*【步骤1】: 将当前帧构造成关键帧*/
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    /*【步骤2】: 将<当前关键帧>设置为当前帧的<参考关键帧>*/
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    /*【步骤3】: 对于双目和RGBD，为当前帧生成新的MapPoints*/
    // 这段代码与updateLastFrame中那一部分代码功能相同
    if(mSensor!=System::MONOCULAR)
    {
        //根据mTcw计算mRcw、mRwc、mtcw、mOw
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
    /*【步骤3.1】: 得到当前帧深度小于阈值的特征点*/
        //创建新的MapPoints,depth < mThDepth(深度图得到的深度比较多，只选择部分添加进MapPoints)
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
    /*【步骤3.2】: 按照深度从小到大排列*/
            sort(vDepthIdx.begin(),vDepthIdx.end());

    /*【步骤3.3】: 将距离比较近的点包装成MapPoints*/
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i); //MapPoints能被哪些关键帧观测到
                    pKF->AddMapPoint(pNewMP,i); //关键帧能观测到哪些MapPoints
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP); //加入到数据库中

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                // 这里决定了双目和RGBD摄像头地图点云的稠密程度
                // 但是仅仅为了地图稠密，直接改这里不好
                // 因为这些MapPoints会参与之后的整个slam过程
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF); //将关键帧插入到localMapping里

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief Tracking::SearchLocalPoints
 * @details 只对除<新增加的>MapPoint进行SearchByProjection，以向mCurrentFrame增加新的MapPoint，用于Track Local Map
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched

    /*【步骤1】: mCurrentFrame.mvpMapPoints是已经在TrackReferenceKeyFrame函数中匹配成功的mapPoints*/
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad()) //用于判断该MapPoint是否因为<能看到它的关键帧数目少于2>而已经被擦出
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; //【迭代】->为了下一步不再search它
                pMP->mbTrackInView = false; //【标志位】【跟踪】: false，不用于跟踪
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    /*【步骤2】: 投影MapPoint，并检查可见性*/
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId) //见，上一步
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
    /*【步骤2.1】: 是否可见*/
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    /*【步骤3】: 只对除<新增加的>MapPoint进行SearchByProjection，以向mCurrentFrame增加新的MapPoint*/
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocated recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th); /** @todo*/
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    // 设置参考MapPoint
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    // 局部关键帧对应的MapPoint就是局部MapPoints
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    // 将keyFrame能观测到的mapPoints都加进去
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief Tracking::UpdateLocalKeyFrames
 * 1、对当前帧中的每个MapPoint为关键帧投票，根据它是否能被观测到
 * 2、把能观测到地图点的keyFrame，放进局部地图中
 * 3、包含那些已经<被包含关键帧>的<邻近关键帧>
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    /*【步骤1】: 对当前帧中的每个MapPoint为关键帧投票，根据它是否能被观测到*/
    // 记录当前帧中MapPoints<被观测到时>所在的<keyFrame>及此<keyFrame出现的次数>
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad()) //we do not currently erase MapPoint from memory
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    /*【步骤2】: 把能观测到地图点的keyFrame，放进局部地图中*/
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad()) /** @todo */
            continue;

        // 记录能观测到当前帧中MapPoints数目最多的关键帧
        if(it->second > max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        //mCurrentFrame的局部keyframe
        mvpLocalKeyFrames.push_back(it->first);
        /** @todo */
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    /*【步骤3】: 包含那些已经<被包含关键帧>的<邻近关键帧>*/
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
    /*【步骤3.1】: 限制局部关键帧的数量，超过80个就不再添加了*/
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

    /*【步骤3.2】: 共同的map point数目设置为10，放进局部地图中*/
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

    /*【步骤3.3】: 获取Childs，放进局部地图中*/
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

    /*【步骤3.3】: 获取Parent，放进局部地图中*/
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    /*【4】: 将与当前帧中匹配到的MapPoints数目最多的关键帧作为mCurrentFrame的mpReferenceKF*/
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();
    //cout << "Relocalization Initiated" << endl;

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    /*【步骤1】: 在重定位中找到与该帧<相似的关键帧>*/
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame); /** @todo */

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    //cout << "Relocalization: candidates =  " << nKFs  << endl;

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    /*【步骤2】: 对每一个候选关键帧进行ORB特征匹配,将其与当前帧之间构造PnP求解器*/
    ORBmatcher matcher(0.75,true);

    //构造PnP solver vector，用于存储PnP solver
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    // 标志位，当匹配点数目小于阈值时抛弃之
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad()) /** @todo*/
            vbDiscarded[i] = true;
        else
        {
            // 通过词袋(BoW)搜索得到匹配点，得到与mCurrentFrame中的keyPoints一样排序的MapPoints
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            // 匹配特征数超过15个,构建PnP进行求解
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]); //构造PnP求解器
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }


    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    /*【步骤3】: 寻找一个有<足够多匹配成功点>支撑的相机位姿*/
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
    /*【步骤3.1】: PnP求解位姿*/
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers); //5次Ransac迭代

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
    /*【步骤3.2】: 当PnP求解成功后，优化位姿*/
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

    /*【步骤3.2.1】: 将<匹配成功的MapPoints>放进mCurrentFrame中,用于后面的Optimizer*/
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }
    /*【步骤3.2.2】: BA优化*/
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
    /*【步骤3.2.3】: 匹配成功的点较少，通过在<一个粗糙的窗口>中执行重映射搜索，再次优化*/
                //【1】如果匹配特征数没有50个，则将候选关键帧对应的map point投影到当前帧继续寻找匹配
                if(nGood<50)
                {
                    //cout << "Relocalization:  inliers < 50 : nGood = " << nGood << endl;

                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        //【2】匹配完成之后再次优化帧
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // 【3】如果匹配数还是没有大于50，则更改投影匹配阈值，再次匹配
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            // 【4】只有匹配的个数大于50，才说明重定位成功，最后对大于50个匹配对的帧再次对帧进行优化
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
    /*【步骤3.2.4】: 匹配成功的点足够多了，重定位成功，跳出大循环，剩下的相似关键帧也不再计算了*/
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        cout << "Relocated" << endl;

        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{
    mpViewer->RequestStop();

    cout << "System Reseting" << endl;
    while(!mpViewer->isStopped())
        usleep(3000);

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
