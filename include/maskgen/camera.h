#pragma once

namespace maskgen {

struct CameraParams {
  int width = 640;
  int height = 480;
  double fx = 500.0;
  double fy = 500.0;
  double cx = 320.0;
  double cy = 240.0;
  double near_plane = 0.01;
  double far_plane = 100.0;
  double eye_x = 0.0;
  double eye_y = 0.0;
  double eye_z = 1.0;
  double target_x = 0.0;
  double target_y = 0.0;
  double target_z = 0.0;
  double up_x = 0.0;
  double up_y = 1.0;
  double up_z = 0.0;
};

}  // namespace maskgen
