# TODO for eoin's PR

- [x] Piper's suggestion for a concrete description of what is required in a `oqupy.gradient` call
- [ ] rewrite description referring to these requirements throughout the text, because this should make everything far clearer and easier to understand
- [ ] Move comparison with FD somewhere other than tutorial
    - [ ] decide where to put it
- [ ] Talk about the requirements for `oqupy.gradient` function

Basically I stole the requirements directly from the compute dynamics, and some of my grievences stem from the way that the requirements for that are talked about. e.g. a process tensor is described as optional, as are `dt`, `num_steps`, but as far as i understand a process tensor is absolutely not optional, and likewise the `dt` and `num_steps` is specified by the PT. 

- [ ] debug for multiple PTs
- [ ] write autograd
- [ ] write optimisation example
- [ ] what to call target state
- [ ] write non-linear target state derivative in tutorial