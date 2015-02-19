package org.opencv.android.services.calibration.camera;

import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.services.CameraView;
import org.opencv.android.services.calibration.CameraCalibrationResult;
import org.opencv.android.services.calibration.CameraInfo;
import org.opencv.core.Mat;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.res.Resources;
import android.hardware.Camera.Size;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.ContextMenu;
import android.view.ContextMenu.ContextMenuInfo;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Toast;

public class CalibrationActivity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "Activity";

    private static final String CALIBRATE_ACTION = CameraCalibrationResult.CALIBRATE_ACTION;

    private CameraInfo mRequestedCameraInfo = null;

    private CameraInfo mCameraInfo = new CameraInfo();
    private CameraCalibrator mCalibrator;

    private CameraView mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            new Handler().postDelayed(new Runnable() {
                                @Override
                                public void run() {
                                    mOpenCvCameraView.enableView();
                                    mOpenCvCameraView.setOnTouchListener(CalibrationActivity.this);
                                }
                            }, 1000);
                        }
                    });
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    protected void sendResponse(CameraCalibrationResult calibrationResult) {
        if (CALIBRATE_ACTION.equals(getIntent().getAction()) && mRequestedCameraInfo != null) {
            Bundle extras = getIntent().getExtras();
            String responseAction = CalibrationActivity.class.getName() + "!response";
            if (extras != null) {
                responseAction = extras.getString("responseAction", responseAction);
            }
            Intent intent = new Intent(responseAction);
            String response = null;
            if (calibrationResult != null) {
                if (mRequestedCameraInfo.equals(calibrationResult.mCameraInfo)) {
                    response = calibrationResult.getJSON().toString();
                }
            }
            if (response != null)
                intent.putExtra("response", response);
            Log.i(TAG, "Send " + (response == null ? "CANCEL" : "VALID") + " response broadcast: " + responseAction);
            sendBroadcast(intent);
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);

        CameraInfo startupCameraInfo = new CameraInfo();
        startupCameraInfo.setPreferredResolution(this);
        if (startupCameraInfo.mWidth > 1280) startupCameraInfo.mWidth = 1280;
        if (startupCameraInfo.mHeight > 720) startupCameraInfo.mHeight = 720;
        if (CALIBRATE_ACTION.equals(getIntent().getAction())) {
            Bundle extras = getIntent().getExtras();
            if (extras != null) {
                startupCameraInfo.readFromBundle(extras);
                mRequestedCameraInfo = startupCameraInfo;
                CameraCalibrationResult result = new CameraCalibrationResult(mRequestedCameraInfo);
                if (extras.getBoolean("force", false) == false && result.tryLoad(this)) {
                    Log.e(TAG, "Return loaded calibration result");
                    sendResponse(result);
                    finish();
                    return;
                }
            } else {
                Log.e(TAG, "No camera info. Ignore invalid request");
                finish();
            }
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.surface_view);
        mOpenCvCameraView = (CameraView) findViewById(R.id.java_surface_view);
        mOpenCvCameraView.setResolution(startupCameraInfo.mWidth, startupCameraInfo.mHeight);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        registerForContextMenu(mOpenCvCameraView);
    }

    @Override
    protected void onStop() {
        sendResponse((mCalibrator == null) ? null : mCalibrator.getCalibrationResult());
        super.onStop();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        String text = Integer.valueOf(width).toString() + "x" + Integer.valueOf(height).toString();
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show();
        if (mCameraInfo.mWidth != width || mCameraInfo.mHeight != height) {
            if (mCalibrator != null)
                Toast.makeText(this, "Camera resolution changed. Recreate calibrator", Toast.LENGTH_LONG).show();
            mCameraInfo = new CameraInfo();
            mCameraInfo.mCameraIndex = mOpenCvCameraView.getCameraIndex();
            mCameraInfo.mWidth = width;
            mCameraInfo.mHeight = height;
            CameraCalibrationResult calibrationResult = new CameraCalibrationResult(mCameraInfo);
            mCalibrator = new CameraCalibrator(mCameraInfo);
            if (calibrationResult.tryLoad(this)) {
                mCalibrator.setCalibrationResult(calibrationResult);
                Toast.makeText(this, "Calibration data loaded from previous launch", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat gray = inputFrame.gray();
        Mat rgba = inputFrame.rgba();
        if (mCalibrator != null)
            mCalibrator.processFrame(gray, rgba);
        return rgba;
    }

    private static final int MENU_GROUP_RESOLUTION = 10;
    private List<Size> mResolutionList;
    private MenuItem[] mResolutionMenuItems;
    private SubMenu mResolutionMenu;
    private static final int MENU_GROUP_CALIBRATOR = 1000;

    @Override
    public void onCreateContextMenu(ContextMenu menu, View v, ContextMenuInfo menuInfo) {
        menu.setHeaderTitle("Camera calibration");

        getMenuInflater().inflate(R.menu.calibration, menu);

        if (mRequestedCameraInfo == null) {
            mResolutionMenu = menu.addSubMenu("Resolution");
            mResolutionList = mOpenCvCameraView.getResolutionList();
            mResolutionMenuItems = new MenuItem[mResolutionList.size()];

            ListIterator<Size> resolutionItr = mResolutionList.listIterator();
            int idx = 0;
            while(resolutionItr.hasNext()) {
                Size element = resolutionItr.next();
                mResolutionMenuItems[idx] = mResolutionMenu.add(MENU_GROUP_RESOLUTION, idx, Menu.NONE,
                        Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
                idx++;
            }
        }

        if (mCalibrator != null) {
            mCalibrator.onCreateMenu(menu, MENU_GROUP_CALIBRATOR);
        }
    }

    @Override
    public boolean onContextItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item.getGroupId() == MENU_GROUP_RESOLUTION)
        {
            int id = item.getItemId();
            Size resolution = mResolutionList.get(id);
            mOpenCvCameraView.setResolution(resolution.width, resolution.height);
            // TODO Fix OpenCV SDK to call this callback automatically
            onCameraViewStarted(resolution.width, resolution.height);
            return true;
        }
        switch (item.getItemId()) {
            case R.id.calibrate:
            {
                if (mCalibrator == null) {
                    return true;
                }

                final Resources res = getResources();
                if (mCalibrator.getCornersBufferSize() < 2) {
                    Toast.makeText(this, res.getString(R.string.more_samples), Toast.LENGTH_SHORT).show();
                    return true;
                }

                new AsyncTask<Void, Void, Void>() {
                    private ProgressDialog calibrationProgress;

                    @Override
                    protected void onPreExecute() {
                        mOpenCvCameraView.disableView();
                        calibrationProgress = new ProgressDialog(CalibrationActivity.this);
                        calibrationProgress.setTitle(res.getString(R.string.calibrating));
                        calibrationProgress.setMessage(res.getString(R.string.please_wait));
                        calibrationProgress.setCancelable(false);
                        calibrationProgress.setIndeterminate(true);
                        calibrationProgress.show();
                    }

                    @Override
                    protected Void doInBackground(Void... arg0) {
                        mCalibrator.calibrate();
                        return null;
                    }

                    @Override
                    protected void onPostExecute(Void result) {
                        calibrationProgress.dismiss();
                        mCalibrator.reset();
                        String resultMessage = (mCalibrator.isCalibrated()) ?
                                res.getString(R.string.calibration_successful) :
                                res.getString(R.string.calibration_unsuccessful);
                        (Toast.makeText(CalibrationActivity.this, resultMessage, Toast.LENGTH_SHORT)).show();

                        if (mCalibrator.isCalibrated()) {
                            CameraCalibrationResult calibrationResult = mCalibrator.getCalibrationResult();
                            calibrationResult.save(CalibrationActivity.this);
                            calibrationResult.saveToStorage(CalibrationActivity.this);
                            if (CALIBRATE_ACTION.equals(CalibrationActivity.this.getIntent().getAction())) {
                                Log.e(TAG, "Return received calibration result");
                                sendResponse(calibrationResult);
                                finish();
                            }
                        }
                        mOpenCvCameraView.enableView();
                    }
                }.execute();
                return true;
            }
            case R.id.save:
            {
                if (mCalibrator == null) {
                    Toast.makeText(CalibrationActivity.this, "Calibrator doesn't exists", Toast.LENGTH_LONG).show();
                    return true;
                }
                CameraCalibrationResult calibrationResult = mCalibrator.getCalibrationResult();
                if (calibrationResult == null) {
                    Toast.makeText(CalibrationActivity.this, "No calibration data for camera!", Toast.LENGTH_LONG).show();
                    return true;
                }
                calibrationResult.saveToStorage(CalibrationActivity.this);
                return true;
            }
        }
        if (mCalibrator != null) {
            if (mCalibrator.onMenuItemSelected(this, item))
                return true;
        }
        return false;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.d(TAG, "onTouch invoked");
        if (event.getAction() == MotionEvent.ACTION_DOWN && mCalibrator != null) {
            mCalibrator.addCorners();
        }
        return false;
    }

    public void onOpenMenuClick(View v) {
        openContextMenu(mOpenCvCameraView);
    }
}
