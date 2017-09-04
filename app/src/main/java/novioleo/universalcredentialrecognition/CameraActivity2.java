package novioleo.universalcredentialrecognition;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.support.annotation.BoolRes;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import static org.opencv.android.Utils.matToBitmap;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.COLOR_RGBA2RGB;
import static org.opencv.imgproc.Imgproc.CV_mRGBA2RGBA;
import static org.opencv.imgproc.Imgproc.Canny;
import static org.opencv.imgproc.Imgproc.RETR_TREE;
import static org.opencv.imgproc.Imgproc.approxPolyDP;
import static org.opencv.imgproc.Imgproc.arcLength;
import static org.opencv.imgproc.Imgproc.contourArea;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.getPerspectiveTransform;
import static org.opencv.imgproc.Imgproc.pyrMeanShiftFiltering;
import static org.opencv.imgproc.Imgproc.warpPerspective;

/**
 * Created by novio on 17-8-31.
 */

public class CameraActivity2 extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static final String    TAG = "OCVSample::Activity";

    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_CANNY    = 2;

    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;
    private int                    mViewMode;
    private MenuItem               mItemPreviewRGBA;
    private MenuItem               mItemPreviewGray;
    private MenuItem               mItemPreviewCanny;
    private OverlayView overlay;
    private Mat roi;
    private Bitmap roi_bitmap;
    private Button click;

    private JavaCameraView mOpenCvCameraView;

    private static Handler handler;
    private HandlerThread handlerThread;

    public CameraActivity2() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);


        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.camera2);

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.activity_surface_view);
        overlay = (OverlayView) findViewById(R.id.roi);
        click = (Button) findViewById(R.id.click);
        click.setEnabled(false);
        mOpenCvCameraView.setFocusable(true);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        return true;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        handlerThread.quit();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            Log.e("UCR","Exception!");
        }
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if(OpenCVLoader.initDebug()){

            mOpenCvCameraView.enableView();

            addCallback(
                    new OverlayView.DrawCallback() {
                        @Override
                        public void drawCallback(Canvas canvas) {
                            renderDebug(canvas);
                        }
                    });
        }

        handlerThread = new HandlerThread("inference"){
            @Override
            public void run() {
                int height = roi.height();
                int width = roi.width();
                Mat tmp = new Mat();
                Mat tmp2 = new Mat();
                pyrMeanShiftFiltering(roi,tmp,3.0,20.0);
                Canny(tmp,tmp2,100,200);
                final List<MatOfPoint> contours = new LinkedList<>();
                findContours(tmp2,contours,new Mat(),RETR_TREE,CHAIN_APPROX_SIMPLE);
                if (contours.size() > 0) {
                    double max_area = contourArea(contours.get(0));
                    MatOfPoint max_area_contours = contours.get(0);
                    for (int i = 1;i < contours.size();++i){
                        double t_area = contourArea(contours.get(i));
                        if (t_area > max_area) {
                            max_area = t_area;
                            max_area_contours = contours.get(i);
                        }
                    }
                    if (max_area >= width*height*0.5){
                        MatOfPoint2f double_max_area_contours =new MatOfPoint2f(max_area_contours.toArray());
                        double peri = arcLength(double_max_area_contours,true);
                        MatOfPoint2f result_curve = new MatOfPoint2f();
                        approxPolyDP(double_max_area_contours,result_curve,0.01*peri,true);
                        Point[] result_points = result_curve.toArray();
                        if (result_points.length == 4){
                            int[] compute_direction = new int[4];
                            for (int i = 0;i < 4;++i){
                                double[] delta = new double[2];
                                delta[0] = result_points[(i+1)%4].x - result_points[i].x;
                                delta[1] = result_points[(i+1)%4].y - result_points[i].y;
                                int delta_argmax = Math.abs(delta[1]) > Math.abs(delta[0]) ? 1 :0;
                                compute_direction[i] = (delta_argmax+1)*(delta[delta_argmax]>-delta[delta_argmax]?1:-1)+2;
                            }
                            Point[] target_points = new Point[]{
                                    new Point(468,300),
                                    new Point(468,0),
                                    new Point(),
                                    new Point(0,300),
                                    new Point(0,0)
                            };
                            Point[] new_target_points = new Point[4];
                            for (int i = 0;i < 4;++i){
                                new_target_points[i] = target_points[compute_direction[i]];
                            }
                            Mat warp_matrix = getPerspectiveTransform(result_curve,new MatOfPoint2f(new_target_points));
                            warpPerspective(roi,tmp,warp_matrix,new Size(468,300));
                            Log.d("UCR","bingo");

                        }
                    }
//                    if (contourArea(contours.get(0)) >= (width*height*0.5)){
//                        handler.sendEmptyMessage(0);
//                    }else{
//                        handler.sendEmptyMessage(1);
//                    }
                }
                roi_bitmap = Bitmap.createBitmap(tmp.width(), tmp.height(), Bitmap.Config.ARGB_8888);
                try {
                    matToBitmap(tmp, roi_bitmap);
                }catch (Exception e){
                    Log.e(TAG,e.getMessage());
                }
            }


        };

        handler = new Handler(){
            @Override
            public void  handleMessage(Message message) {
                switch (message.what){
                    case 0:
                        click.setEnabled(true);
                        break;
                    case 1:
                        click.setEnabled(false);
                        break;
                    default:
                        break;
                }

            }
        };
//        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, getApplicationContext(), mLoaderCallback);
//        System.loadLibrary("opencv_java");
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }



    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    private void addCallback(OverlayView.DrawCallback callback) {
        if (overlay != null) {
            overlay.postInvalidate();
            overlay.addCallback(callback);
        }
    }

    private void renderDebug(Canvas canvas){
        if (overlay != null) {
            overlay.postInvalidate();
        }else{
            return;
        }
        if (roi_bitmap != null) {
            Matrix matrix = new Matrix();
            float scaleFactor = 1.0f;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                    canvas.getWidth() - roi_bitmap.getWidth() * scaleFactor,
                    canvas.getHeight() - roi_bitmap.getHeight() * scaleFactor);
            canvas.drawBitmap(roi_bitmap, matrix, new Paint());
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat rgba = inputFrame.rgba();
        Size frameSize = rgba.size();
        Mat rgb = new Mat();
        Imgproc.cvtColor(rgba,rgb,COLOR_RGBA2RGB);
        Rect rect = new Rect(
                Double.valueOf(0.18*frameSize.width).intValue(),
                0,
                Double.valueOf(frameSize.width*0.4).intValue(),
                Double.valueOf(frameSize.height).intValue()
        );
        roi = new Mat(rgb,rect);
        handlerThread.start();
        return rgb;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemPreviewRGBA) {
            mViewMode = VIEW_MODE_RGBA;
        } else if (item == mItemPreviewGray) {
            mViewMode = VIEW_MODE_GRAY;
        } else if (item == mItemPreviewCanny) {
            mViewMode = VIEW_MODE_CANNY;
        }

        return true;
    }

}
