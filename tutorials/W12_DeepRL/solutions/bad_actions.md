### Question
In the above example we see that action 3 is selected despite action 1 being the better action. Why do you think so? Take a look at the init_logits in the above cell.

### Answer
Although we update policy on actions with better values, we also end up pushing more often on whichever actions happen to have higher values of $π_θ$ to begin with (which could happen due to chance or bad initialization). These actions might end up winning the race to the top in spite of being bad.
