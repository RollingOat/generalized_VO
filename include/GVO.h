#pragma once
// generate a header file
#include <Eigen/Dense>
#include <algorithm>
#include <ceres/ceres.h>
#include <cmath>
#include <iostream>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <vector>

struct control {
  double v;
  double w;
};

struct robot_state {
  double x;
  double y;
  double theta;
};

struct obstacle_state {
  double x;
  double y;
  double vx;
  double vy;
};

struct goal {
  double x;
  double y;
  double theta;
};

struct distanceFunction {
  robot_state robot;
  obstacle_state obs;
  control u;

  typedef double Scalar;
  typedef Eigen::VectorXd InputType;
  typedef Eigen::VectorXd ValueType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType;
  enum {
    InputsAtCompileTime = Eigen::Dynamic,
    ValuesAtCompileTime = Eigen::Dynamic
  };

  int inputs() const { return 1; }
  int values() const { return 1; }
  double operator()(const Eigen::VectorXd &t, Eigen::VectorXd &fvec) const {
    // Example function: f(x, y) = (x-2)^2 + (y+3)^2
    std::vector<double> robot_next_state = {
        robot.x + u.v * cos(robot.theta) * t(0),
        robot.y + u.v * sin(robot.theta) * t(0), robot.theta + u.w * t(0)};
    std::vector<double> obs_next_state = {obs.x + obs.vx * t(0),
                                          obs.y + obs.vy * t(0)};
    double dist = sqrt(pow(robot_next_state[0] - obs_next_state[0], 2) +
                       pow(robot_next_state[1] - obs_next_state[1], 2));
    fvec(0) = dist;
    return dist;
  }
  double operator()(const Eigen::VectorXd &t) const {
    // Example function: f(x, y) = (x-2)^2 + (y+3)^2
    std::vector<double> robot_next_state = {
        robot.x + u.v * cos(robot.theta) * t(0),
        robot.y + u.v * sin(robot.theta) * t(0), robot.theta + u.w * t(0)};
    std::vector<double> obs_next_state = {obs.x + obs.vx * t(0),
                                          obs.y + obs.vy * t(0)};
    double dist = sqrt(pow(robot_next_state[0] - obs_next_state[0], 2) +
                       pow(robot_next_state[1] - obs_next_state[1], 2));
    return dist;
  }
};

struct CostFunctor {
  robot_state robot_;
  obstacle_state obs_;
  control u_;

  template <typename T> bool operator()(const T *const t, T *residual) const {
    std::vector<T> robot_next_state = {
        robot_.x + u_.v * cos(robot_.theta) * t[0],
        robot_.y + u_.v * sin(robot_.theta) * t[0], robot_.theta + u_.w * t[0]};
    std::vector<T> obs_next_state = {obs_.x + obs_.vx * t[0],
                                     obs_.y + obs_.vy * t[0]};
    T dist = sqrt(pow(robot_next_state[0] - obs_next_state[0], 2) + pow(robot_next_state[1] - obs_next_state[1], 2));
    residual[0] = dist;
    return true;
  }

  CostFunctor(robot_state robot, obstacle_state obs, control u) {
    robot_.x = robot.x;
    robot_.y = robot.y;
    robot_.theta = robot.theta;
    obs_.x = obs.x;
    obs_.y = obs.y;
    obs_.vx = obs.vx;
    obs_.vy = obs.vy;
    u_.v = u.v;
    u_.w = u.w;
  }
};

class GeneralizedVO {
private:
  double v_max;
  double w_max;
  double v_step;
  double w_step;
  double t_horizon;
  double t_step;
  double robot_radius;
  double obs_radius;
  double goal_tol;

  double Dt(double dt, const control &u, const obstacle_state &obs,
            const robot_state &robot);
  std::vector<double> find_mininum_time_to_collision(const robot_state &robot,
                                                     const obstacle_state &obs,
                                                     const control &u);

  obstacle_state predict_obstacle_state(const obstacle_state &obs,
                                        const double dt);
  robot_state predict_robot_state(const robot_state &robot, const control &u,
                                  const double dt);

public:
  GeneralizedVO(double v_max, double w_max, double v_step, double w_step,
                double t_horizon, double t_step, double robot_radius,
                double obs_radius, double goal_tol) {
    this->v_max = v_max;
    this->w_max = w_max;
    this->v_step = v_step;
    this->w_step = w_step;
    this->t_horizon = t_horizon;
    this->t_step = t_step;
    this->robot_radius = robot_radius;
    this->obs_radius = obs_radius;
    this->goal_tol = goal_tol;
  }
  control compute_optimal_u(const control &u_desire,
                            const std::vector<control> &u_feasible, const std::vector<double>& d_mins, const std::vector<double>& t_mins);
  std::vector<control>
  compute_feasible_control(const robot_state &robot,
                           const std::vector<obstacle_state> &obstacle_states,
                           const goal &goal_state, std::vector<double>& min_d_mins,
  std::vector<double>& min_t_mins);
  control compute_preferred_control(const robot_state &robot, const double gx,
                                    const double gy, const double gtheta);
};