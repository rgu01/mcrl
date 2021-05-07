/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   mcrl.h
 * Author: ron
 *
 * Created on January 28, 2021, 4:27 PM
 */

#ifndef MCRL_H
#define MCRL_H
#include <iostream>
#include <cassert>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#include <cstring>
#include <cmath>

#define DEFAULT_REWARD 32767

/**
 * Simple implementation of a Q-learning algorithm
 * This implementation is *NOT* intended to be efficient, rather it is
 * implemented in the most straight-forward manner to illustrate how to
 * correctly use the external-learning functionality of Uppaal.
 *
 * Notice that we truncate concrete state-values to nearest integer to avoid
 * and explosion in the Q-table.
 */
class QLearner {
private:
    /**
     * Struct for handling the Q-value update
     */
    struct qvalue_t {
        double _value = 0;
        size_t _count = 0;
    };
    size_t count = 0;
    // type for mapping actions to values
    using qaction_t = std::map<size_t,qvalue_t>;

    // type for mapping states to action-values
    using qstate_t = std::map<std::pair<std::vector<double>,std::vector<double>>,qaction_t>;

    // actual values
    qstate_t _Q;
public:
    // whether we are doing minimization or maximization
    bool _is_minimization = true;
    size_t _d_size = 0; // discrete state-vector size
    size_t _c_size = 0; // continuous state-vector size
    bool learning = true;
    
public:
    /**
     * Converts a raw observation into appropriate format for Q-table
     * @param d_vars
     * @param c_vars
     * @return
     */
    std::pair<std::vector<double>,std::vector<double>> make_state(double* d_vars, double* c_vars) {
        std::vector<double> d_vector;
        std::vector<double> c_vector;
        if(d_vars != nullptr)
        {
            d_vector.resize(_d_size); // make space in vector
            for(size_t d = 0; d < _d_size; ++d) // copy over data
                d_vector[d] = d_vars[d];
        }
        if(c_vars != nullptr)
        {
            c_vector.resize(_c_size); // make space in vector
            for(size_t c = 0; c < _c_size; ++c) // copy over data
            {
                 // truncates to "lump" several concrete states together to avoid a Q-table explosion
                c_vector[c] = std::trunc(c_vars[c]);
            }
        }
        return {d_vector,c_vector};
    }

    /**
     * Returns best known Q-value for the given state (over all actions)
     * @param d_vars
     * @param c_vars
     * @return
     */
    double best_value(double* d_vars, double* c_vars, bool* found) {
        auto state = make_state(d_vars, c_vars);

        // lets try to find a matching state
        auto it = _Q.find(state);
        if(it != _Q.end())
        {
            auto& state_table = it->second;
            double best = std::numeric_limits<double>::infinity();
            if(!_is_minimization)
                best *= -1; // flip to negative infinity if we do maximization
            for(auto& other : state_table)
            {
                if(_is_minimization)
                    best = std::min(best, other.second._value);
                else
                    best = std::max(best, other.second._value);
            }
            *found = true;
            return best;
        }
        else
        {
            *found = false;
            return DEFAULT_REWARD;
        }
    }

public:
    QLearner(bool is_minimization, size_t d_size, size_t c_size) : _is_minimization(is_minimization), _d_size(d_size), _c_size(c_size) {
#ifdef VERBOSE
        std::cerr << "[New Q-Learner (" << this << ") with sizes (" << d_size << ", " << c_size << ") for minimization?=" << std::boolalpha << is_minimization << "]" << std::endl;
#endif
    }

    // this object is default copyable
    QLearner(const QLearner& other) = default;
    

    /**
     * Add an observation/sample. This modifies the Q-values of the given
     * state-vector pair. The value given is assumed to be the delta of the
     * observed cost/reward between the (d_vars,c_vars) pair to the (t_d_vars,t_c_vars)
     * pair when the action was used.
     * Notice that t_d_vars and t_c_vars may be null if the terminal state was
     * reached (i.e. a unique sink-state with a permanent q-value of zero).
     * @param d_vars discrete values of current state
     * @param c_vars continuous values of current state
     * @param action action used
     * @param t_d_vars discrete values of next state
     * @param t_c_vars continuous values of next state
     * @param value is the immediate reward/cost
     */
    
    void add_sample(double* d_vars, double* c_vars, size_t action, double* t_d_vars, double* t_c_vars, double reward)
    {
        bool found = false;
        const double gamma = 0.99; // discount, we could make it converge to zero by making this dependent on the number of samples seen for this state-action-pair
        const double alpha = 2.0; // constant learning rate
        auto from_state = make_state(d_vars, c_vars);
        auto future_estimate = best_value(t_d_vars, t_c_vars, &found);
        qvalue_t& q = _Q[from_state][action];
        const double learning_rate = 1.0/std::min<double>(alpha,q._count+1);
        //const double learning_rate = 1.0/alpha;
        assert(learning_rate <= 1.0);
        if(q._count == 0)
        {
            // special case, we have no old value
            q._value = learning_rate * reward;
        }
        else
        {
            // standard Q-value update
            q._value = q._value + learning_rate * (reward + gamma * future_estimate - q._value);
        }
        q._count += 1;
    }

    /**
     * Returns the Q-value for a given action (a) or the lowest (resp highest if maximization)
     * value of any other action (a') observed if no observation has yet been made
     * of the action (a).
     * @param d_vars
     * @param c_vars
     * @param action
     * @return
     */
    double value(double* d_vars, double* c_vars, size_t action, bool* found) {
        auto state = make_state(d_vars, c_vars);

        // lets try to find a matching state
        auto it = _Q.find(state);
        if(it != _Q.end())
        {
            // we have observed this state before
            auto& state_table = it->second;
            auto action_it = state_table.find(action);
            if(action_it != state_table.end())
            {
                *found = true;
                // we have observations for this action, return the computed Q-value
                if(action_it->second._value != DEFAULT_REWARD)
                {
                    return action_it->second._value;
                }
                else
                {
                    return action_it->second._value + 1;
                }
            }
            else
            {
                // No prior observation of the action, we need a default value.
                *found = false;
                return DEFAULT_REWARD;
                /*if(printed)
                {
                    return DEFAULT_REWARD;
                }
                else
                {
                    return best_value(d_vars, c_vars, &found);
                }*/
            }
        }
        else
        {
            // No prior observation of the state, we need a default value.
            *found = false;
            return DEFAULT_REWARD;
        }
    }

    /**
     * Inspects whether the given action is the "best" for the given state.
     * I.e. if we minimize, it will be the action with the lowest Q-value.
     * Several actions can be equally good.
     * @param d_vars
     * @param c_vars
     * @param action
     * @return
     */
    bool is_allowed(double* d_vars, double* c_vars, size_t action, bool* found) 
    {
        //bool stop = false;
        bool is_current_found = false;
        bool is_best_found = false;
        double current_v = value(d_vars, c_vars, action, &is_current_found);
        double best_v = best_value(d_vars, c_vars, &is_best_found);
        auto state = make_state(d_vars, c_vars);// this is for debug
        //std::ostream& out = std::cerr;// this is for debug
        
        // when print is called, learning turns false
        if(!learning)
        {
            if(!is_best_found && action != 0)
            {
                // this is when the model goes to a state that is not in the Q-table
                // and action ID is not 0.
                //stop = true; 
                /*out << "(";
                for(size_t d = 0; d < d_size(); ++d)
                {
                    out << d_vars[d] << ",";
                }
                out << "): " << action << " is not found. \n";*/
            }
            else if(action == 0)
            {
                // this is when the model goes to a state that is not in the Q-table
                // and action ID is 0.
                //stop = true;
            }
        }
        *found = is_current_found & is_best_found;
        
        if(*found && current_v == best_v)
        {
            return true;
        }
        else if(!is_current_found)
        {
            // if the current state and action is not found, 
            // then the action is allowed for exploration
            return true; 
        }
        else if(!is_best_found)
        {
            // if the state is not in the Q-table so the best action of the state is not found
            // then the action is not 
            return true; 
        }
        
        return false;
    }
    
    int length()
    {
        if(&_Q != nullptr) return _Q.size();
        else return 0;
    }

    size_t d_size()
    {
        return _d_size;
    }
    
    /**
     * Outputs the learned q-values to the string-stream in a json-friendly format
     * of a map over "(discrete,continuous)"-state variable vector pairs and
     * into maps from actions to q-values.
     * @param out - the output stream to write to, defaults to stderror.
     */
    void print(std::ostream& out = std::cerr) {
        bool first = true;
        learning = false;
        for(auto& state_action : _Q) {
            auto& state = state_action.first;
            if(!first) out << ",\n"; // make json-friendly
            first = false;
            out << "\"(";
            // iterate over discrete state values
            for(auto& d_value : state.first)
            {
                out << d_value << ",";
            }
            out << "),[";
            // iterate over concrete/continuous state values
            for(auto& c_value : state.second)
            {
                out << c_value << ",";
            }
            out << "]\":{";
            auto& action_map = state_action.second;
            bool first_action = true;
            for(auto& action_value : action_map)
            {
                if(!first_action) out << ",";
                first_action = false;
                out << "\n\t";
                //action_value.first is the action ID.
                //action_value.second is the value of the state-action pair, and the count of it.
                out << "\"" << action_value.first << "\":" << action_value.second._value;
            }
            out << "}";
        }
        
        //out << "length: " << _Q.size();
    }
};

#endif /* MCRL_H */

