
                            ONE-STEP ONLINE ACTOR-CRITIC ALGORITHM PSEUDOCODE

  gamma  = 0.9  #Discount factor 
  
  for i in epochs:
  
      state = environment.get_state()  #getting the first  state from the environment
  
      value = critic(state)            # computing value function's prediction of current state.
      policy = actor(state)            # computing the probability distribution of the current state.
      action = policy.sample()         # choosing action a from the probability distribution A.
  
      next_state, reward  = environment.take_action(action)  # computing the next state and the reward R for taking the particular action a.
      next_value = critic(next_state)                        # computing the value prediction for next state taken after initial state.
  
      advantage = reward + (gamma*next_value - value)        # computing the advantage as reward R + (discounted value of next state - value of previous state.) 
  
      NB: In online learning the Advantage can also be called the Temporal Difference Error(TD Error), its worth noting that the Advantange
      formula would differ for Monte Carlo Actor-Critic Algorithms
  
      loss_actor = - policy.log_prob(action) * advantage            # Reinforces the action that was just taken based on the advantage.
    loss_critic = advantage.pow(2)                                # typical online critic update (MSE on TD error)

    update_actor(loss_actor)
    update_critic(loss_critic)

   
#The loop being online (one transition at a time, update immediately).
#TD error in online learning is valid for the TD(0) Actor-Critic update, but not universally true for all advantage estimators (e.g., GAE, n-step returns, etc.).
