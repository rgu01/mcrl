#include "mcrl.h"

// we use this to check that we do not deallocate an object twice.
// this should never happen, so this is *only* useful if you suspect that
// Uppaal Stratego is doing something wrong.
std::set<QLearner*> live;
//int count1 = 0,count2 = 0;

/**
 * Allocates an instance of a learner
 * @param minimization, flag for determining optimization type (minimization=true/maximization=false)
 * @param d_size, size of the discrete array
 * @param c_size, size of the continuous array
 * @param a_size, number of (controllable) actions available in the system
 * @return a pointer to a learner object
 */
extern "C" void* uppaal_external_learner_alloc(bool minimization, size_t d_size, size_t c_size, size_t a_size) {
    auto object = new QLearner(minimization, d_size, c_size);
    live.insert(object); // for later sanitycheck
    return object;
}

int count = 0;
/**
 * Deallocation code for objects allocated by uppaal_external_learner_alloc
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 */
extern "C" void uppaal_external_learner_dealloc(void* object) {
    QLearner* obj = (QLearner*) object;
    std::cerr << count++ << ": Q-table's length: " << obj->length() << "\n";
    //obj->reduce();
    if (obj != nullptr && live.count(obj) != 1) {
        assert(false && "Call-sequence from UPPAAL was wrong, please report to the UPPAAL developers");
    }

    delete obj;
    live.erase(obj);
    return;
}

/**
 * Write the state of the learner (called by saveStrategy in uppaal)
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 */
extern "C" char* uppaal_external_learner_print(void* object) {
    std::stringstream outstream;
    QLearner* ql = (QLearner*) object;
    ql->print(outstream); // ask learning object to be printed to a stream
    auto data = outstream.str(); // convert the stream into a regular string object
    char* tmp = new char[data.size()+1]; // create a c-style string with enough space
    strcpy(tmp, data.c_str()); // copy over the data
    return tmp; // deallocation is handled by the caller
}

/**
 * Deep-copy function of an instance of a leaner.
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 * @return a pointer to a duplicate/deep-copy of object
 */
extern "C" void* uppaal_external_learner_clone(void* object) {
    assert(object != nullptr);
    auto new_object = new QLearner(*(QLearner*) object);
    live.insert(new_object);
    return new_object;
}

/**
 * Called for each sample in a trace. Given a trace on s_0-a->s_1-b-> .. s_n
 * samples a received in inverse-order (s_1-b->s_2 is seen before s_0-a->s_1)
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 * @param action, the action taken
 * @param from_d_vars, the discrete state-vector of the origin state
 * @param from_c_vars, the continuous state-vector of the origin state
 * @param t_d_vars, the discrete state-vector of the target state
 * @param t_c_vars, the continuous state-vector of the target state
 * @param value, the observed cost/reward (see @uppaal_external_learner_alloc, minimization)
 */
extern "C" void uppaal_external_learner_sample_handler(void* object, size_t action,
        double* from_d_vars, double* from_c_vars,
        double* t_d_vars, double* t_c_vars, double value) 
{
    //bool stop = true;
    if (object == nullptr) {
        return;
    }
    auto q = (QLearner*) object;
    /*auto from_state = q->make_state(from_d_vars,from_c_vars);
    auto to_state = q->make_state(t_d_vars, t_c_vars);
    if(from_d_vars[0] == 3 && from_d_vars[6] == 3)
    {
        stop = true;
    }*/
    q->add_sample(from_d_vars, from_c_vars, action, t_d_vars, t_c_vars, value/100);
}

extern "C" void uppaal_external_learner_online_sample_handler(void* object, size_t action,
        double* from_d_vars, double* from_c_vars,
        double* t_d_vars, double* t_c_vars, double value) {
    /*if (object == nullptr) {
        return;
    }
    auto q = (QLearner*) object;
    q->add_sample(from_d_vars, from_c_vars, action, t_d_vars, t_c_vars, value);*/
}

/**
 * Function for returning the result of the leaner; used both during training (is_eval=false)
 * and evaluation (is_eval=true)
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 * @param is_eval, indicating whether we are evaluating or training
 * @param action, indicating the action taken from the state where d_vars and t_vars were observed
 * @param d_vars, the observed discrete state-vector
 * @param t_vars, the observed continuous state-vector
 */
extern "C" double uppaal_external_learner_predict(void* object, bool is_eval, size_t action, double* d_vars, double* c_vars) {
    // you can control search here!
    // return ONLY weights > 0, non inf and non nan.
    // a weighted choice will be done over all actions according to the weight
    auto q = (QLearner*) object;
    //std::ostream& out = std::cerr;
    bool found = false, isBest = false;
    double d_value = 1.0;
    
    if(is_eval)
    {
        isBest = q->is_allowed(d_vars, c_vars, action, &found);
        if(isBest && found)
        {
            d_value = 1.0;
        }
        else
        {
            d_value = 0.0;
        }
    }
    else
    {
        //d_value = 1.0;
        //d_value = 1e100 - q->value(d_vars, c_vars, action, &found);
        d_value = 1.0 - q->value(d_vars, c_vars, action, &found);
        //assert(d_value >= 0);
    }
    
    return d_value;
}

/**
 * Batch-completion call-back
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 */
extern "C" void uppaal_external_learner_flush(void* object) {
    // not used by q-learning
    return;
}

