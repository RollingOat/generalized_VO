#include "GVO.h"
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <queue>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <vector>
#include <visualization_msgs/Marker.h>

class vo_node {
private:
  ros::NodeHandle nh_;
  double v_max;
  double w_max;
  double v_step;
  double w_step;
  double t_horizon;
  double t_step;
  double robot_radius;
  double obs_radius;
  double goal_tol;
  GeneralizedVO gvo;
  goal goal_state;
  bool goal_received;
  std::queue<std::vector<obstacle_state>> obs_states_queue;
  // Publishers
  ros::Publisher twist_pub_;
  // subscribers to topics of type PoseStamped and Odometry and ModelStates
  ros::Subscriber goal_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber obs_states_sub_;
  ros::Publisher marker_pub;
  ros::Subscriber real_obs_states_sub_;

public:
  vo_node(const ros::NodeHandle &nh, double v_max, double w_max, double v_step,
          double w_step, double t_horizon, double t_step, double robot_radius,
          double obs_radius, double goal_tol)
      : nh_(nh), gvo(v_max, w_max, v_step, w_step, t_horizon, t_step,
                    robot_radius, obs_radius, goal_tol), goal_received(false) {
    this->v_max = v_max;
    this->w_max = w_max;
    this->v_step = v_step;
    this->w_step = w_step;
    this->t_horizon = t_horizon;
    this->t_step = t_step;
    this->robot_radius = robot_radius;
    this->obs_radius = obs_radius;
    this->goal_tol = goal_tol;
    twist_pub_ = nh_.advertise<geometry_msgs::Twist>("/scarab40/cmd_vel", 10);
    // subscribers to topics of type PoseStamped and Odometry and ModelStates
    goal_sub_ = nh_.subscribe("/scarab40/move_base_simple/goal", 10, &vo_node::goalCallback, this);
    odom_sub_ = nh_.subscribe("/scarab40/ground_truth/odom", 10, &vo_node::odomCallback, this);
    obs_states_sub_ =
        nh_.subscribe("/gazebo/model_states", 10, &vo_node::obsStateCallback, this);
    real_obs_states_sub_ =
        nh_.subscribe("/scarab40/obs_states", 10, &vo_node::realObsStateCallback, this);
    marker_pub = nh_.advertise<visualization_msgs::Marker>(
        "visualization_marker", 10);
  }

  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr &msg) {
    goal_state.x = msg->pose.position.x;
    goal_state.y = msg->pose.position.y;
    goal_state.theta = msg->pose.orientation.z;
    goal_received = true;
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    if (!goal_received) {
      ROS_INFO("Goal not received yet");
      return;
    }
    robot_state rs;
    rs.x = msg->pose.pose.position.x;
    rs.y = msg->pose.pose.position.y;
    // compute theta using quaternion
    double x = msg->pose.pose.orientation.x;
    double y = msg->pose.pose.orientation.y;
    double z = msg->pose.pose.orientation.z;
    double w = msg->pose.pose.orientation.w;
    rs.theta = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    if(sqrt(pow(goal_state.x - rs.x, 2) + pow(goal_state.y - rs.y, 2)) < goal_tol){
        ROS_INFO("Goal reached");
    }
    std::vector<obstacle_state> lastest_obs_states = obs_states_queue.back();
    ROS_INFO("Current goal is (%f, %f)", goal_state.x, goal_state.y);
    ROS_INFO("Current robot state is (%f, %f, %f)", rs.x, rs.y, rs.theta);
    ros::Time start = ros::Time::now();
    std::vector<double> d_mins;
    std::vector<double> t_mins;
    std::vector<control> u_feasible =
        gvo.compute_feasible_control(rs, lastest_obs_states, goal_state, d_mins, t_mins);
    ROS_INFO("u_feasible size: %d", u_feasible.size());
    control u_desire = gvo.compute_preferred_control(
        rs, goal_state.x, goal_state.y, goal_state.theta);
    ROS_INFO("u_desire: (%f, %f)", u_desire.v, u_desire.w);
    control u_optimal = gvo.compute_optimal_u(u_desire, u_feasible, d_mins, t_mins);
    ROS_INFO("u_optimal: (%f, %f)", u_optimal.v, u_optimal.w);
    ros::Time end = ros::Time::now();
    ROS_INFO("Predicted next robot state is: (%f, %f, %f)", rs.x + u_optimal.v * cos(rs.theta) * t_step, rs.y + u_optimal.v * sin(rs.theta) * t_step, rs.theta + u_optimal.w * t_step);
    ROS_INFO("time to find optimal u: %f", (end - start).toSec());
    ROS_INFO("\n");
    geometry_msgs::Twist twist;
    twist.linear.x = u_optimal.v;
    twist.angular.z = u_optimal.w;
    twist_pub_.publish(twist);
    // visualize the velocity
    visualization_msgs::Marker marker;
    marker.header.frame_id = "scarab40/odom";
    marker.header.stamp = ros::Time();
    marker.ns = "scarab40";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = rs.x;
    marker.pose.position.y = rs.y;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    double theta = rs.theta;
    marker.pose.orientation.z = sin(theta / 2);
    marker.pose.orientation.w = cos(theta / 2);
    marker.scale.x = u_optimal.v;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker_pub.publish(marker);
  }

  void obsStateCallback(const gazebo_msgs::ModelStates::ConstPtr &msg) {
    std::vector<obstacle_state> obs_states;
    for (int i = 0; i < msg->name.size(); i++) {
      if (msg->name[i].find("actor") != std::string::npos) {
        obstacle_state obs;
        obs.x = msg->pose[i].position.x;
        obs.y = msg->pose[i].position.y;
        obs.vx = msg->twist[i].linear.x;
        obs.vy = msg->twist[i].linear.y;
        obs_states.push_back(obs);
      }
    }
    obs_states_queue.push(obs_states);
    if (obs_states_queue.size() > 10) {
      obs_states_queue.pop();
    }
  }

  void realObsStateCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    obstacle_state obs;
    obs.x = msg->pose.pose.position.x;
    obs.y = msg->pose.pose.position.y;
    obs.vx = msg->twist.twist.linear.x;
    obs.vy = msg->twist.twist.linear.y;
    std::vector<obstacle_state> obs_states;
    obs_states.push_back(obs);
    obs_states_queue.push(obs_states);
    if (obs_states_queue.size() > 10) {
      obs_states_queue.pop();
    }
  }
};

int main(int argc, char **argv) {
  // Initialize ROS
  ros::init(argc, argv, "gvo_node");
  ros::NodeHandle nh;
  double v_max = 1;
  double w_max = 1.5;
  double v_step = 0.1;
  double w_step = 0.3;
  double t_horizon = 2;
  double t_step = 0.1;
  double robot_radius = 0.5;
  double obs_radius = 0.5;
  double goal_tol = 0.5;
  vo_node vo(nh, v_max, w_max, v_step, w_step, t_horizon, t_step, robot_radius,
             obs_radius, goal_tol);
  // Loop
  ros::Rate rate(10); // Adjust the loop rate as needed
  while (ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
