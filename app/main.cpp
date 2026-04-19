#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/imgcodecs.hpp>

static void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program << " [OPTIONS] <mesh_file>\n\n"
            << "Options:\n"
            << "  -o, --output PATH      Output PNG path (default: mask.png)\n"
            << "  --width INT            Image width (default: 640)\n"
            << "  --height INT           Image height (default: 480)\n"
            << "  --fx FLOAT             Focal length x (default: 500.0)\n"
            << "  --fy FLOAT             Focal length y (default: 500.0)\n"
            << "  --cx FLOAT             Principal point x (default: width/2)\n"
            << "  --cy FLOAT             Principal point y (default: height/2)\n"
            << "  --near FLOAT           Near plane (default: 0.01)\n"
            << "  --far FLOAT            Far plane (default: 100.0)\n"
            << "  --eye X Y Z            Camera position (default: 0 0 1)\n"
            << "  --target X Y Z         Look-at target (default: 0 0 0)\n"
            << "  --up X Y Z             Up vector (default: 0 1 0)\n"
            << "  --mesh-tx FLOAT        Mesh translation x (default: 0)\n"
            << "  --mesh-ty FLOAT        Mesh translation y (default: 0)\n"
            << "  --mesh-tz FLOAT        Mesh translation z (default: 0)\n"
            << "  --mesh-rx FLOAT        Mesh rotation x in degrees (default: 0)\n"
            << "  --mesh-ry FLOAT        Mesh rotation y in degrees (default: 0)\n"
            << "  --mesh-rz FLOAT        Mesh rotation z in degrees (default: 0)\n"
            << "  -h, --help             Show this help\n";
}

static bool ParseVector3(const std::vector<std::string>& args, size_t* index,
                         double* x, double* y, double* z) {
  if (*index + 3 >= args.size()) {
    return false;
  }
  try {
    *x = std::stod(args[*index + 1]);
    *y = std::stod(args[*index + 2]);
    *z = std::stod(args[*index + 3]);
    *index += 3;
  } catch (...) {
    return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);

  std::string mesh_path;
  std::string output_path = "mask.png";
  maskgen::CameraParams params;
  maskgen::MeshPose pose;

  const double deg_to_rad = 3.14159265358979323846 / 180.0;

  for (size_t i = 1; i < args.size(); ++i) {
    const auto& arg = args[i];

    if (arg == "-h" || arg == "--help") {
      PrintUsage(args[0].c_str());
      return 0;
    } else if (arg == "-o" || arg == "--output") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing output path\n";
        return 1;
      }
      output_path = args[i];
    } else if (arg == "--width") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing width value\n";
        return 1;
      }
      params.width = std::stoi(args[i]);
    } else if (arg == "--height") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing height value\n";
        return 1;
      }
      params.height = std::stoi(args[i]);
    } else if (arg == "--fx") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing fx value\n";
        return 1;
      }
      params.fx = std::stod(args[i]);
    } else if (arg == "--fy") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing fy value\n";
        return 1;
      }
      params.fy = std::stod(args[i]);
    } else if (arg == "--cx") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing cx value\n";
        return 1;
      }
      params.cx = std::stod(args[i]);
    } else if (arg == "--cy") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing cy value\n";
        return 1;
      }
      params.cy = std::stod(args[i]);
    } else if (arg == "--near") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing near value\n";
        return 1;
      }
      params.near_plane = std::stod(args[i]);
    } else if (arg == "--far") {
      if (++i >= args.size()) {
        std::cerr << "Error: missing far value\n";
        return 1;
      }
      params.far_plane = std::stod(args[i]);
    } else if (arg == "--eye") {
      if (!ParseVector3(args, &i, &params.eye_x, &params.eye_y, &params.eye_z)) {
        std::cerr << "Error: --eye requires 3 values\n";
        return 1;
      }
    } else if (arg == "--target") {
      if (!ParseVector3(args, &i, &params.target_x, &params.target_y, &params.target_z)) {
        std::cerr << "Error: --target requires 3 values\n";
        return 1;
      }
    } else if (arg == "--up") {
      if (!ParseVector3(args, &i, &params.up_x, &params.up_y, &params.up_z)) {
        std::cerr << "Error: --up requires 3 values\n";
        return 1;
      }
    } else if (arg == "--mesh-tx") {
      if (++i >= args.size()) return 1;
      pose.tx = std::stod(args[i]);
    } else if (arg == "--mesh-ty") {
      if (++i >= args.size()) return 1;
      pose.ty = std::stod(args[i]);
    } else if (arg == "--mesh-tz") {
      if (++i >= args.size()) return 1;
      pose.tz = std::stod(args[i]);
    } else if (arg == "--mesh-rx") {
      if (++i >= args.size()) return 1;
      pose.rx = std::stod(args[i]) * deg_to_rad;
    } else if (arg == "--mesh-ry") {
      if (++i >= args.size()) return 1;
      pose.ry = std::stod(args[i]) * deg_to_rad;
    } else if (arg == "--mesh-rz") {
      if (++i >= args.size()) return 1;
      pose.rz = std::stod(args[i]) * deg_to_rad;
    } else if (arg[0] == '-') {
      std::cerr << "Error: unknown option: " << arg << "\n";
      return 1;
    } else {
      mesh_path = arg;
    }
  }

  if (mesh_path.empty()) {
    std::cerr << "Error: mesh file path is required\n";
    PrintUsage(args[0].c_str());
    return 1;
  }

  if (params.cx == 320.0 && params.width != 640) {
    params.cx = params.width / 2.0;
  }
  if (params.cy == 240.0 && params.height != 480) {
    params.cy = params.height / 2.0;
  }

  maskgen::Mesh mesh;
  if (!mesh.LoadFromFile(mesh_path)) {
    std::cerr << "Error: failed to load mesh: " << mesh_path << "\n";
    return 1;
  }

  std::cout << "Mesh loaded: " << mesh.vertices().size() / 3 << " vertices, "
            << mesh.indices().size() / 3 << " triangles\n";

  try {
    maskgen::MaskGenerator generator(params);
    cv::Mat mask = generator.Generate(mesh, pose);

    if (!cv::imwrite(output_path, mask)) {
      std::cerr << "Error: failed to write output image: " << output_path << "\n";
      return 1;
    }

    std::cout << "Mask saved to: " << output_path
              << " (" << mask.cols << "x" << mask.rows << ")\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
