/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	if (is_initialized)
		return;
	
	num_particles = 100;

	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);
	
	for (int i = 0; i < num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;
		
		particles.push_back(particle);
		weights.push_back(1);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	for (int i = 0; i < num_particles; i++)
	{
		double new_x;
		double new_y;
		double new_theta;
		
		if (fabs(yaw_rate) < EPS) {
			new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			new_theta = particles[i].theta;
		} else{
			new_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate*delta_t;
		}
		
		normal_distribution<double> N_x(new_x, std_pos[0]);
	    normal_distribution<double> N_y(new_y, std_pos[1]);
	    normal_distribution<double> N_theta(new_theta, std_pos[2]);
		
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	//IMP: Use the dist helper function
	double euclidean_dist;
	
	for (int i = 0 ; i < observations.size(); i++) {
		double minimum_dist = numeric_limits<double>::max();
        int minimum_id = -1000;		
		for (int j = 0; j < predicted.size(); j++) {
			euclidean_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (euclidean_dist < minimum_dist) {
				minimum_dist = euclidean_dist;
				minimum_id = predicted[j].id;
			}
		}
		observations[i].id = minimum_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();

	for (int i = 0; i < particles.size(); ++i) {
		std::vector<LandmarkObs> trans_observations;
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs trans_obs;
			trans_obs.x = observations[j].x * cos(particles[i].theta) -
					observations[j].y * sin(particles[i].theta) + particles[i].x;
			trans_obs.y = observations[j].x * sin(particles[i].theta) +
					observations[j].y * cos(particles[i].theta) + particles[i].y;
			trans_obs.id = -1;
			trans_observations.push_back(trans_obs);
		}

		std::vector<LandmarkObs> predicted;
		
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			double distance_p_obs;
			distance_p_obs = dist(particles[i].x, particles[i].y,
					map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if (distance_p_obs <= sensor_range) {
				LandmarkObs obs;
				obs.id = map_landmarks.landmark_list[j].id_i;
				obs.x = map_landmarks.landmark_list[j].x_f;
				obs.y = map_landmarks.landmark_list[j].y_f;
				predicted.push_back(obs);
			}
		}

		dataAssociation(predicted, trans_observations);

		double prob_mul = 1.0;
		double prob;
		int check_idx;
		for (int j = 0; j < predicted.size(); ++j) {
			for (int k = 0; k < trans_observations.size(); ++k) {
				 if (predicted[j].id == trans_observations[k].id) {
					check_idx = k;
					prob = exp(-((predicted[j].x - trans_observations[check_idx].x) *
						(predicted[j].x - trans_observations[check_idx].x) /
						(2 * std_landmark[0] * std_landmark[0]) +
						(predicted[j].y - trans_observations[check_idx].y) *
						(predicted[j].y - trans_observations[check_idx].y) /
						(2 * std_landmark[1] * std_landmark[1]))) /
						(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
						
						prob_mul = prob_mul * prob;
				}
			}
		}
    
		weights.push_back(prob_mul);
		particles[i].weight = prob_mul;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	discrete_distribution<int> distribution(weights.begin(), weights.end());

    vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; ++i)
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}
	
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
	
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
