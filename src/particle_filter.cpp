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
	
	num_particles = 400;

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
		// Transform observations to MAP's coordinate system
		std::vector<LandmarkObs> observations_map;
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs obs;
			obs.x = observations[j].x * cos(particles[i].theta) -
					observations[j].y * sin(particles[i].theta) + particles[i].x;
			obs.y = observations[j].x * sin(particles[i].theta) +
					observations[j].y * cos(particles[i].theta) + particles[i].y;
			obs.id = -1;
			observations_map.push_back(obs);
		}

		// Compute predicted measurements
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

		dataAssociation(predicted, observations_map);

		double prob = 1.0;
		double prob_i;
		for (int j = 0; j < predicted.size(); ++j) {
			double min_dist = numeric_limits<double>::max();
			int min_idx = -1;
			for (int k = 0; k < observations_map.size(); ++k) {
			// Use measurement closest to predicted
				if (predicted[j].id == observations_map[k].id) {
					double check_dist = dist(predicted[j].x, predicted[j].y,
									observations_map[k].x, observations_map[k].y);
					if (check_dist < min_dist) {
						min_dist = check_dist;
						min_idx = k;
					}
				}
			}
			if (min_idx != -1) {
				prob_i = exp(-((predicted[j].x - observations_map[min_idx].x) *
						(predicted[j].x - observations_map[min_idx].x) /
						(2 * std_landmark[0] * std_landmark[0]) +
						(predicted[j].y - observations_map[min_idx].y) *
						(predicted[j].y - observations_map[min_idx].y) /
						(2 * std_landmark[1] * std_landmark[1]))) /
						(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
				prob = prob * prob_i;
			}
		}
    
		weights.push_back(prob);
		particles[i].weight = prob;
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
