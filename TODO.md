# TODO for eoin's PR

- [x] Piper's suggestion for a concrete description of what is required in a `oqupy.gradient` call
- [ ] rewrite description referring to these requirements throughout the text, because this should make everything far clearer and easier to understand
- [ ] Punctuate equations in tutorial
- [ ] Move comparison with FD somewhere other than tutorial
    - [ ] decide where to put it
- [ ] Talk about the requirements for `oqupy.gradient` function
- [ ] Find the missing factor of -1 in gradient
    - [x] Fix fact target DM was not transposed
    - [ ] Decide should it be transposed in backend or frontend
- [ ] Fix the gradient of just the numbers $\Theta_0$ and $\Delta$ for ARP

Basically I stole the requirements directly from the compute dynamics, and some of my grievences stem from the way that the requirements for that are talked about. e.g. a process tensor is described as optional, as are `dt`, `num_steps`, but as far as i understand a process tensor is absolutely not optional, and likewise the `dt` and `num_steps` is specified by the PT. 

- [ ] debug for multiple PTs
- [ ] write autograd
- [ ] write optimisation example
- [ ] what to call target state
- [ ] write non-linear target state derivative in tutorial