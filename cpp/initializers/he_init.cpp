#include "he_init.h"
#include <iostream>

using namespace eddl;
using namespace std;

layer he_normal(layer l, int seed){
    //delete l->init;
    l->init=new IHeNormal(seed);
    return l;
}


/**
 * He normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in )
 * where fan_in is the number of input units in the weight tensor 
 *
 * @param seed int; Used to seed the random generator.
*/
IHeNormal::IHeNormal(int seed) : Initializer("he_normal") {
    // Todo: Implement
    this->seed = seed;
}

void IHeNormal::apply(Tensor* params)
{
    if (params->ndim == 1)
	params->rand_signed_uniform(0.1f);
    else if (params->ndim == 2) {
      	params->rand_normal(0.0f, ::sqrtf(2.0f / (params->shape[0])));
    }
    else if (params->ndim == 4) {// only fan_in
       	int rf=params->shape[2]*params->shape[3];
      	int fin=rf*params->shape[1];
      	params->rand_normal(0.0f, ::sqrtf(2.0f / (float)(fin)));
      	// params->rand_normal(0.0f, ::sqrtf(2.0f / ((float)params->size / params->shape[0])));
    } 
    else {
      	params->rand_signed_uniform(0.1f);
    }
}

