# Poximal Polocy Optimization (PPO)

    - rl algo using a on-policy method
    - on-policy method means the algo learn a policy to make decsions in the enviorment

# How it works at a high level

    - first collect trajectories
        - agent takes an action, enviorment returns a trajectory (state, action, reward, next_state)
    - next compute advantage estimates
        - advantage function computes how much better an action is compared to the average action at that state
        - PPO uses Genralized Advantage Estimation (GAE)
    - next update the policy
        - PPO uses a special objective function (to prevent the policy from updating too much in on episode, ensuring stability)
        - implmented by adding a penalty to the objective function if the new policy deviates too much from the original policy
    - iterate!

# Whats good about PPO

    - the objective function used for policy updates
    - uses a "clipped" verions of the policy ratio, adding a penalty if the new policy deviates too much
    - this ensures stability and effcient learning
