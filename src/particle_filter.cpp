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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 100;

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	particles = vector<Particle>(num_particles);
	for(int i=0;i<num_particles;++i){
		particles[i].id = i;
		particles[i].weight = 1;

		// Initialize with radom Gaussian noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for (int i = 0; i < num_particles; ++i) {
		double x_n,y_n,theta_n;


		double yaw_rate_1=yaw_rate;
		//Avoiding that yaw_rate could be equal to null
		if(yaw_rate_1 == 0){
			yaw_rate_1 += 0.00000001;
		}

		// Predicting the new position
		x_n = particles[i].x +velocity / yaw_rate_1*(sin(particles[i].theta + yaw_rate_1*delta_t) - sin(particles[i].theta));
		y_n = particles[i].y +velocity / yaw_rate_1*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_1*delta_t));
		theta_n = particles[i].theta + yaw_rate_1*delta_t;

		// Create normal distributions for x, y and theta
		normal_distribution<double> dist_x(x_n, std_pos[0]);
		normal_distribution<double> dist_y(y_n, std_pos[1]);
		normal_distribution<double> dist_theta(theta_n, std_pos[2]);

		particles[i].x = dist_x(gen);

		//  add random Gaussian noise to the new PC positions
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); ++i){
		double dis_min = 100, dis = 0;
		int ind_min = 0;

		int j=0;
		while(predicted[j].id>0){
			// computing the Euclidean distance
			dis = sqrt( pow(observations[i].x - predicted[j].x,2) + pow(observations[i].y - predicted[j].y,2));
			if(dis<dis_min){
				// Actualize the actual minimum distance and its index
				dis_min = dis;
				ind_min = j;
			}
			j++;
		}
		// Assign the closest Predicted measurement id  to the observation
		observations[i].id= predicted[ind_min].id;
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

	std::vector<LandmarkObs> observations_m,observations_p;

	observations_m = vector<LandmarkObs>(observations.size());
	observations_p = vector<LandmarkObs>(5*observations.size());

	//Init observation prediction id with 0;
	for(int i=0;i<observations_p.size();++i){
		observations_p[i].id = 0;
	}
	//cout<<"***Update start"<<endl;w
	// Updating the particle weights
	for(int i=0;i<num_particles;++i){
		// Particle coordinates
		double xp,yp,theta;
		xp = particles[i].x;
		yp = particles[i].y;
		theta = particles[i].theta;

		// Convert all observations
		for(int j =0;j<observations.size();++j){
			// observation coordinates in particle space
			double xc,yc;
			xc = observations[j].x;
			yc = observations[j].y;
			// Convert observation
			observations_m[j].x = xp + cos(theta)*xc - sin(theta)*yc;
			observations_m[j].y = yp + sin(theta)*xc + cos(theta)*yc;
		}
		//Finding the landmarks in the nearby of the particle
		int l =0;
		for(int j =0;j<map_landmarks.landmark_list.size();++j){
			// Landmark coordinates
			double xl,yl,dis;
			xl = map_landmarks.landmark_list[j].x_f;
			yl = map_landmarks.landmark_list[j].y_f;
			dis = sqrt(pow(xp-xl,2)+pow(yp-yl,2));
			if(dis<=sensor_range){
				//cout<<"Close landmarks IDs: "<<map_landmarks.landmark_list[j].id_i<<endl;
				observations_p[l].id = map_landmarks.landmark_list[j].id_i;
				observations_p[l].x = map_landmarks.landmark_list[j].x_f;
				observations_p[l].y = map_landmarks.landmark_list[j].y_f;
				l++;
			}
		}
		//data association
		dataAssociation(observations_p,observations_m);
		// Calculating weight
		for(int j=0;j<observations_m.size();++j){
			double gauss_norm,exponent;
			double xo,yo,xpr,ypr;
			xo = observations_m[j].x;
			yo = observations_m[j].y;
			//cout<<"Map ID: "<<observations_m[j].id-1<<endl;
			xpr = map_landmarks.landmark_list[observations_m[j].id-1].x_f;
			ypr = map_landmarks.landmark_list[observations_m[j].id-1].y_f;

			gauss_norm = (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));
			exponent = pow((xo - xpr),2)/(2 * pow(std_landmark[0],2)) + pow((yo - ypr),2)/(2 * pow(std_landmark[1],2));
			particles[i].weight = particles[i].weight*exp(-exponent);
		}
		}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	std::discrete_distribution<> dd(0,num_particles);
	int index;
	index = dd(gen);
	double beta = 0.0;
	// nomalizing weights:
	double sum_w=0;
	for(int i=0;i<num_particles;++i){
		sum_w += particles[i].weight;
	}
	for(int i=0;i<num_particles;++i){
		particles[i].weight/=sum_w;
	}

	// finding max weight;
	double maxw = particles[0].weight;
	for(int i=1;i<num_particles;++i){
		if(particles[i].weight>maxw){
			maxw = particles[i].weight;
		}
	}
	for (int i=0;i<num_particles;++i){
		beta += rand() / double(RAND_MAX)*2*maxw;
		while (beta>particles[index].weight){
			beta -= particles[index].weight;
			index = (index + 1)%num_particles;
		}
		particles[i] = particles[index];
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
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
