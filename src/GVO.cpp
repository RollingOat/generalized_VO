// implement the generalized velocity obstacle class in c++ based on it python
// version

#include "GVO.h"

control GeneralizedVO::compute_optimal_u(const control &u_desire,
                                         const std::vector<control> &u_feasible,
                                         const std::vector<double> &d_min,
                                         const std::vector<double> &t_min) {
  // compute the optimal control based on the feasible control and the desired
  // control input: u_desire: the desired control
  //        u_feasible: the feasible control
  // output: u_optimal: the optimal control

  control u_optimal;
  double best_d = -1;
  double best_t = -1;
  double min_cost = 1000000;
  for (int i = 0; i < u_feasible.size(); i++) {
    double cost = pow(u_feasible[i].v - u_desire.v, 2) +
                  pow(u_feasible[i].w - u_desire.w, 2);
    if (cost < min_cost) {
      min_cost = cost;
      u_optimal.v = u_feasible[i].v;
      u_optimal.w = u_feasible[i].w;
      best_d = d_min[i];
      best_t = t_min[i];
    }
  }
  std::cout << "           best_d: " << best_d << std::endl;
  std::cout << "           best_t: " << best_t << std::endl;
  return u_optimal;
}

std::vector<control> GeneralizedVO::compute_feasible_control(
    const robot_state &robot,
    const std::vector<obstacle_state> &obstacle_states, const goal &goal_state,
    std::vector<double> &min_d_mins, std::vector<double> &min_t_mins) {
  // compute the feasible control based on the robot state, obstacle state and
  // goal state input: robot: the robot state
  //        obstacle_states: the obstacle states
  //        goal_state: the goal state
  // output: u_feasible: the feasible control
  // std::cout<< "DEBUG!!!! (1)" << std::endl;
  std::vector<control> u_feasible;

  for (double v = -v_max; v < v_max + v_step; v += v_step) {
    for (double w = -w_max; w < w_max + w_step; w += w_step) {
      control u;
      u.v = v;
      u.w = w;
      std::vector<double> d_mins;
      std::vector<double> t_mins;
      for (int i = 0; i < obstacle_states.size(); i++) {
        // std::cout<< "DEBUG!!!! (2)" << std::endl;
        std::vector<double> res =
            find_mininum_time_to_collision(robot, obstacle_states[i], u);
        // std::cout<< "DEBUG!!!! (3)" << std::endl;
        t_mins.push_back(res[0]);
        d_mins.push_back(res[1]);
      }
      auto min_d_min_iter = std::min_element(d_mins.begin(), d_mins.end());
      double min_d_min = *min_d_min_iter;
      double min_t_min = t_mins[min_d_min_iter - d_mins.begin()];
      //  std::cout<< "min_d_min: " << min_d_min << std::endl;
      // std::cout<< "min_t_min: " << min_t_min << std::endl;
      if (min_d_min > robot_radius + obs_radius &&
          (min_t_min > t_step || abs(min_t_min) < 0.001)) {
        u_feasible.push_back(u);
        min_d_mins.push_back(min_d_min);
        min_t_mins.push_back(min_t_min);
      }
    }
  }
  return u_feasible;
}

double GeneralizedVO::Dt(double dt, const control &u, const obstacle_state &obs,
                         const robot_state &robot) {
  // compute the time to collision
  // input: dt: the time step
  //        u: the control
  //        obs: the obstacle state
  //        robot: the robot state
  // output: Dt: the time to collision
  robot_state rs = predict_robot_state(robot, u, dt);
  obstacle_state os = predict_obstacle_state(obs, dt);
  double dx = rs.x - os.x;
  double dy = rs.y - os.y;
  double distance = sqrt(pow(dx, 2) + pow(dy, 2));
  return distance;
}

std::vector<double> GeneralizedVO::find_mininum_time_to_collision(
    const robot_state &robot, const obstacle_state &obs, const control &u) {
  // std::vector<double> res;
  // distanceFunction df;
  // df.robot = robot;
  // df.obs = obs;
  // df.u = u;
  // Eigen::VectorXd x(1);
  // x(0) = 0;
  // Eigen::NumericalDiff<distanceFunction> ndf(df);
  // Eigen::LevenbergMarquardt<Eigen::NumericalDiff<distanceFunction>, double>
  // lm(ndf); lm.minimize(x);

  // res.push_back(x(0));
  // res.push_back(df(x));
  // return res;
  // std::cout<< "DEBUG!!!! (4)" << std::endl;
  std::vector<double> res;
  ceres::Problem problem;
  double t_init = 0;
  problem.AddParameterBlock(&t_init, 1);
  problem.SetParameterLowerBound(&t_init, 0, 0);
  problem.SetParameterUpperBound(&t_init, 0, t_horizon);
  // std::cout<< "DEBUG!!!! (5)" << std::endl;
  ceres::CostFunction *cost_functor =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(
          new CostFunctor(robot, obs, u));
  // std::cout<< "DEBUG!!!! (6)" << std::endl;
  problem.AddResidualBlock(cost_functor, nullptr, &t_init);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  // std::cout<< "DEBUG!!!! (7)" << std::endl;
  ceres::Solve(options, &problem, &summary);
  // std::cout<< "DEBUG!!!! (8)" << std::endl;
  res.push_back(t_init);
  // std::cout<< "DEBUG!!!! (9)" << std::endl;
  res.push_back(Dt(t_init, u, obs, robot));
  // std::cout<< "DEBUG!!!! (10)" << std::endl;
  return res;
}

control GeneralizedVO::compute_preferred_control(const robot_state &robot,
                                                 const double gx,
                                                 const double gy,
                                                 const double gtheta) {
  // compute the preferred control based on the robot state and the goal state
  // input: robot: the robot state
  //        gx: the goal x
  //        gy: the goal y
  //        gtheta: the goal theta
  // output: u_preferred: the preferred control
  control u_preferred;
  if (sqrt(pow(gx - robot.x, 2) + pow(gy - robot.y, 2)) < goal_tol) {
    u_preferred.v = 0;
    u_preferred.w = 0;
    return u_preferred;
  }

  double goal_theta = atan2(gy - robot.y, gx - robot.x);
  double theta_diff = 0;
  if (goal_theta <= 0 && robot.theta <= 0) {
    theta_diff = goal_theta - robot.theta;
  } else if (goal_theta > 0 && robot.theta > 0) {
    theta_diff = goal_theta - robot.theta;
  } else if (goal_theta > 0 && robot.theta < 0) {
    theta_diff = goal_theta - robot.theta;
    if (theta_diff >= M_PI) {
      theta_diff = theta_diff - 2 * M_PI;
    }
  } else if (goal_theta < 0 && robot.theta > 0) {
    theta_diff = goal_theta - robot.theta;
    if (theta_diff <= -M_PI) {
      theta_diff = theta_diff + 2 * M_PI;
    }
  }
  // std::cout<<"theta_diff: "<<theta_diff<<std::endl;
  //   if (theta_diff / t_step > w_max)
  //     u_preferred.w = w_max;
  //   else if (theta_diff / t_step < -w_max)
  //     u_preferred.w = -w_max;
  //   else
  //     u_preferred.w = theta_diff / t_step;
  u_preferred.w = theta_diff / t_step;
  if (abs(theta_diff) > 0.01) {
    u_preferred.v = 0;
    return u_preferred;
  }
//   if(abs(theta_diff) < 5.0/180.0*M_PI){
//       u_preferred.w = theta_diff / t_step /5.0;
//       return u_preferred;
//   }
  // if(abs(theta_diff) < w_max*t_step){
  //     u_preferred.v = v_max;
  //     u_preferred.w = 0;
  //     return u_preferred;
  // }
  u_preferred.v = v_max;

  return u_preferred;
}

obstacle_state GeneralizedVO::predict_obstacle_state(const obstacle_state &obs,
                                                     const double dt) {
  // predict the obstacle state based on the obstacle state and the time step
  // input: obs: the obstacle state
  //        dt: the time step
  // output: obs_pred: the predicted obstacle state
  obstacle_state obs_pred;
  obs_pred.x = obs.x + obs.vx * dt;
  obs_pred.y = obs.y + obs.vy * dt;
  obs_pred.vx = obs.vx;
  obs_pred.vy = obs.vy;
  return obs_pred;
}

robot_state GeneralizedVO::predict_robot_state(const robot_state &robot,
                                               const control &u,
                                               const double dt) {
  // predict the robot state based on the robot state, the control and the time
  // step input: robot: the robot state
  //        u: the control
  //        dt: the time step
  // output: robot_pred: the predicted robot state
  robot_state robot_pred;
  robot_pred.x = robot.x + u.v * cos(robot.theta) * dt;
  robot_pred.y = robot.y + u.v * sin(robot.theta) * dt;
  robot_pred.theta = robot.theta + u.w * dt;
  return robot_pred;
}
