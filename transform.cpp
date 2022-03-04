//
//  transform.cpp
//  openCV
//
//  Created by Yifan Huang on 2022-02-04.
//

#include <stdio.h>
#include "transform.hpp"

#define ANGLE_GAP 1
#define ANGLE_MAX 180
#define THRESHOLD 50
#define KERNEL_SIZE 3

void polarToCartesian(const int& rho, const int& theta, Point& p1, Point& p2){
    double a = cos(theta), b = sin(theta);
    double x0 = cvRound(rho*cos(theta));
    double y0 = cvRound(rho*sin(theta));
    p1.x = cvRound(x0 + 1000*(-b));
    p1.y = cvRound(y0 + 1000*(a));
    p2.x = cvRound(x0 - 1000*(-b));
    p2.y = cvRound(y0 - 1000*(a));
}

Mat* hough_Space(const Mat& src){
    CV_Assert(!src.empty());

    Mat *src_buff = new Mat(), *edge_Map = new Mat();
    *src_buff = src.clone();
    if(src.channels() == 3){
        cvtColor(*src_buff, *src_buff, COLOR_BGR2GRAY);
    }
   
    //take maximum distance from two sides
    int distance_Max = hypot(src_buff->rows, src_buff->cols);
    vector<vector<int>>* votes = new vector<vector<int>>(2*distance_Max, vector<int>(ANGLE_MAX+1,0));
    //get edge map
    blur(*src_buff, *src_buff, Size(3,3));
    Canny(*src_buff, *edge_Map, THRESHOLD, THRESHOLD*4, KERNEL_SIZE);
    //debug(*edge_Map);
    int rho, theta;
    for(int i =0; i<edge_Map->rows; i++){
        for(int j=0; j<edge_Map->cols; j++){
            //at the condition of edge point
            if(edge_Map->at<uint8_t>(i,j) == 255){
                for(theta=0; theta<=ANGLE_MAX; theta+=ANGLE_GAP){
                    rho = round(i*sin(theta-90) + j*cos(theta-90))+distance_Max;
                    //rho = round(i*cos(theta) + j*sin(theta))+distance_Max;//not working
                    votes->at(rho).at(theta)++;
                }
            }
        }
    }
    int r=2*distance_Max, c=ANGLE_MAX+1;
    Mat *out = new Mat(r, c, CV_64F);
    for(unsigned i=0;i<r;i++){
        for(unsigned j=0;j<c;j++){
            out->at<double>(i,j) = votes->at(i).at(j);
        }
    }
    delete src_buff;
    delete edge_Map;
    return out;
}

Mat* hough_Transform(const Mat& src, const int& lineThreshold){
    CV_Assert(!src.empty());

    Mat *src_buff = new Mat(), *edge_Map = new Mat(), *lines = new Mat();
    *src_buff = src.clone();
    if(src.channels() == 3){
        cvtColor(*src_buff, *src_buff, COLOR_BGR2GRAY);
    }
   
    //take maximum distance from two sides
    int distance_Max = hypot(src_buff->rows, src_buff->cols);
    vector<vector<int>>* votes = new vector<vector<int>>(2*distance_Max, vector<int>(ANGLE_MAX+1,0));
    
    //get edge map
    blur(*src_buff, *src_buff, Size(3,3));
    Canny(*src_buff, *edge_Map, THRESHOLD, THRESHOLD*4, KERNEL_SIZE);
    
    //debug(*edge_Map);
    
    int rho, theta;
    for(int i =0; i<edge_Map->rows; i++){
        for(int j=0; j<edge_Map->cols; j++){
            //at the condition of edge point
            if(edge_Map->at<uint8_t>(i,j) == 255){
                for(theta=0; theta<=ANGLE_MAX; theta+=ANGLE_GAP){
                    rho = round(i*sin(theta-90) + j*cos(theta-90))+distance_Max;
                    //rho = round(i*cos(theta) + j*sin(theta))+distance_Max;//not working
                    votes->at(rho).at(theta)++;
                }
            }
        }
    }
    
    Point p1, p2;
    cvtColor(*edge_Map, *lines, COLOR_GRAY2BGR);
    //find the max votes
    for(int i=0; i<votes->size(); i++){
        for(int j=0; j<votes->at(i).size(); j++){
            if(votes->at(i).at(j) > lineThreshold){
                rho = i-distance_Max;
                theta = j-90;
                
                printf("find the line with rho at %d and theta at %d \n", rho, theta);
                
                polarToCartesian(rho, theta, p1, p2);
                
                //call line method
                line(*lines, p1, p2, Scalar(0,0,255), 1, LINE_AA);
            }
        }
    }
    
    //add dummy point
    //Point dummyA(10,10), dummyB(100,100);
    //line(*lines, dummyA, dummyB, Scalar(0,0,255),1,LINE_AA);
 
    for(auto &p: {src_buff,edge_Map}){
        delete p;
    }
    
    return lines;
}


//calculate Ix^2 or Iy^2
Mat* gradiant_Square(const Mat& src){
    CV_Assert(!src.empty());
    Mat* buf_01 = new Mat(src.size(),src.type());
    
    //use 64 bit float
    if(src.depth()!=CV_64F){
        src.convertTo(*buf_01, CV_64F);
    }
    else{
        src.copyTo(*buf_01);
    }
    
    Mat *buf_02 = new Mat();
    multiply(*buf_01, *buf_01, *buf_02);
    delete buf_01;
   
    return buf_02;
}

//calculate Ix*Iy
Mat* gradiant_IxIy(const Mat& src1, const Mat& src2){
    CV_Assert(!src1.empty() && !src2.empty());
    Mat* buf_01 = new Mat(src1.size(),src1.type());
    Mat* buf_02 = new Mat(src2.size(),src2.type());
    
    //use 64 bit float
    if(src1.depth()!=CV_64F){
        src1.convertTo(*buf_01, CV_64F);
    }
    else{
        src1.copyTo(*buf_01);
    }
    if(src2.depth()!=CV_64F){
        src2.convertTo(*buf_02, CV_64F);
    }
    else{
        src2.copyTo(*buf_02);
    }
    Mat* buf_out = new Mat();
    multiply(*buf_01, *buf_02, *buf_out);
    delete buf_01;
    delete buf_02;
    
    return buf_out;
}

//operate on single channel source
Mat* harri_Corner_Detect(const Mat& srcX, const Mat& srcY, double a){
    //windows size chosen to be 5 X 5
    CV_Assert(!srcX.empty() && !srcY.empty());
    CV_Assert(srcX.rows == srcY.rows);
    Mat *srcX_ch = new Mat();
    Mat *srcY_ch = new Mat();
    if(srcX.channels()==3){
        extractChannel(srcX, *srcX_ch, 0);
    }else{
        srcX.copyTo(*srcX_ch);
    }
    if(srcY.channels()==3){
        extractChannel(srcY, *srcY_ch, 0);
    }else{
        srcY.copyTo(*srcY_ch);
    }
    
    Mat *Respon = new Mat(srcX.size(), CV_64F);
    copyMakeBorder(*Respon, *Respon, 2, 2, 2, 2, BORDER_REPLICATE);
    
    Mat *grad1 = new Mat(), *grad2 = new Mat(), /*gradA = new Mat(), *gradB = new Mat()*/
    *gradX = new Mat(), *gradY = new Mat();
    srcX_ch->copyTo(*grad1); srcY_ch->copyTo(*grad2);
    //add padding
    copyMakeBorder(*grad1, *grad1, 2, 2, 2, 2, BORDER_REPLICATE);
    copyMakeBorder(*grad2, *grad2, 2, 2, 2, 2, BORDER_REPLICATE);
    
    if(grad1->depth() != CV_64F){
        grad1->convertTo(*gradX, CV_64F);
    }
    else{
        grad1->copyTo(*gradX);
    }
    if(grad2->depth() != CV_64F){
        grad2->convertTo(*gradY, CV_64F);
    }
    else{
        grad2->copyTo(*gradY);
    }
    //debug(*gradX);
    //gradX = normalize_One(*gradA);
    //gradY = normalize_One(*gradB);
    //double sum1=summary(*grad1);
    //double sum2=summary(*grad2);
    //*gradX = *grad1 * (1/sum1);
    //*gradY = *grad2 * (1/sum2);
    //debug(*gradX);
    //cout << *gradX << endl;
    //calculate the determinant of M : lend1 * lenda2
    for(int i=2; i<Respon->rows-2; i++){
        for(int j=2; j<Respon->cols-2; j++){
            //calculate X gradiant square
            Mat winX = (*gradX)(Range(i-2,i+3), Range(j-2, j+3));
            Mat *Ix2 = gradiant_Square(winX);
            double a11 = summary(*Ix2);
            //cout << winX <<endl;
            //calculate Y gradiant square
            Mat winY = (*gradY)(Range(i-2,i+3), Range(j-2, j+3));
            Mat *Iy2 = gradiant_Square(winY);
            double a22 = summary(*Iy2);
            
            //calculate X gradiant * Y gradiant
            Mat *Ixy = gradiant_IxIy(winX, winY);
            double a12 = summary(*Ixy);
            double a21 = a12;
            
            double detM = a11*a22 - a21*a12;
            double traceM = a11 + a22;
            
            //R=det(M)-k(trace(m)^2
            double pixel = detM - (a * traceM * traceM);
            //if (pixel > 65025){
            //    pixel = 65025;
            //}
            //if(pixel < -65025){
            //    pixel = -65025;
            //}
            Respon->at<double>(i,j) = pixel;
            //cout << Respon->at<double>(i,j) <<endl;
            delete Ix2; delete Iy2; delete Ixy;
        }
    }
    //cout << *Respon << endl;
    //debug(*Respon);
    //remove padding
    Mat *Respon_out1 = new Mat();;
    //Respon_out1 = new Mat(*Respon, Range(2, Respon->rows-2), Range(2, Respon->cols-2));
    remove_Padding(*Respon, 2).copyTo(*Respon_out1);
    //debug(*Respon_out1);
    //normalize to [0,1]
    //Mat *Respon_out2 = new Mat();
    Mat *Respon_out2 = new Mat(*Respon_out1);
    //Respon_out2 = normalize_One(*Respon_out1);
   
    Respon_out1->convertTo(*Respon_out2, CV_32F);
    //normalize_twoFF(Respon_out2);
    //threshold_R(Respon_out2, 250);
    //debug(*Respon_out2);
    
    for(auto &p:{grad1,grad2,gradX,gradY,Respon,Respon_out1,srcX_ch,srcY_ch}){
        delete p;
    }
    return Respon_out2;
}

//normalize matrix to [0,255]
void normalize_twoFF (Mat *src){
    CV_Assert(!src->empty());
    if(src->depth() != CV_64F){
        src->convertTo(*src, CV_64F);
    }
    double *minMaX = find_MinMax(*src);
    if(minMaX[0] > 0){//min value > 0
        double sum1 = summary(*src);
        for(int i=0; i<src->rows; i++){
            for(int j=0; j<src->cols; j++){
                src->at<double>(i,j) = src->at<double>(i,j) / sum1 * 255;
            }
        }
    }
    else{//min value < 0
        for(int i=0; i<src->rows; i++){
            for(int j=0; j<src->cols; j++){
                src->at<double>(i,j) = (src->at<double>(i,j)-minMaX[0]) / (minMaX[1]-minMaX[0]) * 255;
                //src->at<double>(i,j) = (src->at<double>(i,j)-minMaX[0]);
            }
        }
    }
}

//thresohld the image
void threshold_R (Mat *src, const int& threshold){
    CV_Assert(!src->empty());
    if(src->depth() != CV_64F){
        src->convertTo(*src, CV_64F);
    }
    for(int i=0; i<src->rows; i++){
        for(int j=0; j<src->cols; j++){
            if(src->at<double>(i,j) < threshold){
                src->at<double>(i,j) = 0;
            }
        }
    }
}

