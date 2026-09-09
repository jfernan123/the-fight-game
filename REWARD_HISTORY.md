# Reward function history

Every version of the walker's reward, why it changed, and what broke it. GitHub renders the LaTeX
below natively (`$...$` inline, `$$...$$` block).

The through-line: **almost every version was defeated by the policy finding something the reward
measured badly**, and each fix came from measuring the actual behavior rather than guessing. The
failures are more informative than the formulas, so they're recorded here too.

---

## v0 — Raw forward velocity

$$r = v_x - c_{\text{energy}} \sum_i a_i^2$$

The obvious first try: go forward, don't waste energy.

**How it broke.** $v_x$ has no ceiling, so "faster" was *always* better with no downside until the
crash. Measured over a single episode: forward velocity climbed monotonically $0 \rightarrow 5.3$
m/s with no plateau, including a 25-step (0.25 s) stretch with both feet off the ground. It learned
to leap and crash, not to walk.

---

## v1 — Bounded velocity tracking + shaping terms

Adopted the shaping *pattern* from Rudin et al., *Learning to Walk in Minutes*
([arXiv:2109.11978](https://arxiv.org/abs/2109.11978)) Appendix A.3 — many small weighted terms
under one bounded objective. Their weights are tuned for a quadruped, so only the structure carried
over, not the numbers.

$$r_{\text{track}} = \exp\!\left(-\frac{(v^{*} - v_x)^2}{\sigma}\right), \qquad \sigma = 0.25$$

$$r = r_{\text{track}} - w_{y}v_y^2 - w_{z}v_z^2 - w_{\omega}\lVert\omega\rVert^2 - w_{\ddot q}\lVert\ddot q\rVert^2 - w_{\dot q}\lVert\dot q\rVert^2 - w_{\Delta a}\lVert a - a_{\text{prev}}\rVert^2 + r_{\text{air}} - P_{\text{fall}}$$

Now bounded at $1.0$, so speed alone can't run away.

**Feet air-time term**, paid once per touchdown:

$$r_{\text{air}} = \sum_{f \in \{r,l\}} \mathbb{1}[\text{landed}_f]\; w_{\text{air}}\left(\min(t_f, t_{\text{cap}}) - t^{*}\right)$$

**How it broke.** A tuning disaster first: $w_{\ddot q} = 10^{-5}$ looked small, but raw joint
accelerations here reach $3{,}000$–$11{,}000$ rad/s², so squared that term alone was $-140$ to
$-1760$ per step — swamping everything else by 3–4 orders of magnitude. Measured per-term and
dropped it to $2\times10^{-9}$.

---

## v2 — Anti-bounding penalty

The policy started bounding (both feet airborne much of the time) because hitting the speed target
that way scored the same as walking.

$$r \mathrel{-}= w_{\text{both air}} \cdot \mathbb{1}[\text{neither foot on ground}]$$

**Removed almost immediately** — *"if jumping is a valid strategy then let it be it."* Kept here
because v6 revisits the same problem from a different angle.

---

## v3 — Orientation penalty

$$r \mathrel{-}= w_{\text{orient}}\left(\theta_{\text{pitch}}^2 + \theta_{\text{roll}}^2\right)$$

Nothing rewarded *staying upright*: tracking only reads $v_x$, so a controlled forward topple that
converts gravity into speed scored identically to a real gait, right up to the fall threshold.

**How it broke — a measurement bug, not a formula bug.** $\theta$ was read from the *root torso
joint*. Once `waist_bend` became actuated (it sits between the root and the chest), that stopped
meaning what it says. Measured on a real checkpoint:

| quantity | value |
|---|---|
| root pitch (what the penalty saw) | within $\pm 30°$ all episode — "upright" |
| **chest pitch (physical reality)** | **$-49.8°$**, at the $50°$ fall threshold |
| cost paid | $0.0098$/step instead of $0.418$/step — **43× too cheap** |

The policy had found a posture the penalty structurally could not see. Fixed by measuring the
chest's true world orientation from its rotation matrix. The same blind spot existed in the angular
*velocity* term and was fixed the same way.

---

## v4 — Gait phase clock + foot slip

Air time shapes *how long* a foot is up, never *which* foot should be down *when*, so a shuffle or
an asymmetric hop scored as well as a real alternating walk. Added a clock of period
$T_{\text{gait}} = 0.8$ s, with the right foot expected planted for the first half:

$$r_{\text{gait}} = w_{\text{gait}} \sum_{f}\Big(\mathbb{1}\big[c_f = c_f^{*}(\phi)\big] - \mathbb{1}\big[c_f \neq c_f^{*}(\phi)\big]\Big)$$

The policy observes $\sin 2\pi\phi, \cos 2\pi\phi$ (not raw $\phi$, to avoid the wraparound
discontinuity).

**Slip penalty**, against a planted foot skating:

$$r \mathrel{-}= w_{\text{slip}} \sum_f \mathbb{1}[c_f]\left(v_{f,x}^2 + v_{f,y}^2\right)$$

**Two bugs found here.**

*Inconsistent targets:* $t^{*} = 0.5$ s (a quadruped value) while the clock implies
$T_{\text{gait}}/2 = 0.4$ s — so a foot obeying the schedule perfectly was paid a **negative**
air-time bonus for it. Now derived as $t^{*} = T_{\text{gait}}/2$ so they can't drift apart.

*Wrong velocity:* `cvel`'s linear half is the velocity at the **subtree centre of mass**, not at the
foot, so it picks up a large $\omega \times r$ lever-arm term. Measured: it reported **1.6–1.75
m/s** of slip for a foot genuinely moving **0.13–0.20 m/s** — an 8–10× overestimate, 60–100× once
squared, making slip the single largest ongoing cost *even standing still* ($-0.42$/step). Fixed by
shifting the reference point to the body's inertial centre:

$$v_{\text{foot}} = v_{\text{com}} + \omega \times \left(x_{\text{ipos}} - x_{\text{com ref}}\right)$$

verified to match `mj_objectVelocity` exactly (0.0 error).

---

## v5 — Randomized command, balance, alive bonus

**Randomized command.** A fixed $v^{*}$ let a canned motion specialize on one number — a single
lunge per cycle tracks 1.5 m/s fine. Now resampled per episode and every 5 s:

$$v^{*} \sim \mathcal{U}(0.5,\, 2.0)\ \text{m/s}$$

$v^{*}$ is part of the observation — without seeing it the task is unsolvable.

**Balance.** Nothing measured *keep your weight over your feet*, which is what balance actually
means — you can be perfectly upright, both feet down, and still toppling:

$$r \mathrel{-}= w_{\text{bal}} \left\lVert \text{CoM}_{xy} - \tfrac{1}{2}\left(p_{r,xy} + p_{l,xy}\right) \right\rVert^2$$

**Alive bonus.** $r \mathrel{+}= 0.1$ per step.

**The bug this fixed was the worst one.** With $\sigma = 0.25$, a stationary agent against a 1.5 m/s
command scores $\exp(-9) = 0.0001$ — the only positive term in the entire reward was unreachable
until you were *already* moving ~1 m/s. Every other term being a penalty, **total per-step reward
was strictly negative, making "terminate immediately" the optimal policy.** Observed exactly that:
`ep_len_mean` pinned flat at **9.4 steps**. Fixed by widening $\sigma: 0.25 \rightarrow 1.0$ and
adding the alive bonus.

> A separate, non-reward bug contributed: mjwarp's contact buffer is fixed-size and **not cleared
> between steps** — only the first `d.nacon` entries are valid. Filtering on `dist <= 0` let
> **93 of 96 stale slots** through as live contacts, so the "non-foot touching ground" check fired
> constantly and killed episodes regardless of the policy.

---

## v6 — Air-time cap (current)

$$r \mathrel{-}= w_{\text{excess}} \sum_f \max\!\left(0,\; t_f - t_{\max}\right), \qquad t_{\max}=0.45\ \text{s},\ w_{\text{excess}}=2.0$$

The landing bonus prices *touchdowns*; nothing stopped a foot from simply staying up. Measured on
the first genuinely-walking policy: both feet off the ground **40.3%** of all steps — bounding, not
walking (a human walk is ~0%). Unlike v2 this doesn't forbid leaving the ground, it only bites past
a normal swing, so brief flight stays free.

### ⚠️ Status: measured, and it does not work

After 52M steps of fine-tuning with this term:

| | before (798M) | after (850M) |
|---|---|---|
| median flight | 0.140 s | 0.160 s |
| p90 flight | 0.530 s | 0.530 s |
| flights over $t_{\max}$ | 37.0% | 34.0% |
| both feet airborne | 41.0% | 37.5% |
| `reward/feet_air_time` | $-0.0236$ | $-0.0218$ |

Unchanged within noise. **The weight is too small to matter**: at $-0.022$/step it is ~2% of the
$+0.94$ total, so trading any speed (worth $0.86$) for shorter flights is a bad deal. The policy is
correctly ignoring an underspecified penalty.

**Proposed:** raise $w_{\text{excess}}: 2.0 \rightarrow 20.0$ and lower
$t_{\max}: 0.45 \rightarrow 0.30$ s. Not yet applied — and v7 below explains why the policy may
have been *unable* to respond regardless of the weight.

---

## v7 — Action-magnitude barrier (current)

$$r \mathrel{-}= w_{\text{sat}} \cdot \frac{1}{n}\sum_i \tilde{a}_i^2, \qquad w_{\text{sat}} = 0.001$$

where $\tilde{a}$ is the **pre-clamp** action — what the policy commanded — as opposed to
$a = \mathrm{clip}(\tilde{a}, -1, 1)$, which is what physics receives and what every other effort
term measures.

**The problem.** Because energy cost was applied to the *post-clamp* action, commanding $24$ cost
exactly what commanding $1$ cost. Nothing opposed the network's raw output growing without bound,
and it grew for the entire run:

| step | mean $\lvert a\rvert$ | saturated | actor out-layer $\lVert W\rVert$ | $\log\sigma$ |
|---|---|---|---|---|
| 2M | 0.22 | 0.8% | 0.045 | +0.009 |
| 104M | 1.74 | 61.5% | 0.070 | −0.582 |
| 208M | 6.10 | 70.2% | 0.126 | +0.209 |
| 416M | 15.85 | 82.0% | 0.259 | +1.080 |
| 936M | **24.19** | **83.1%** | 0.377 | **+1.271** |

A $110\times$ growth in commanded magnitude, ending with 83% of joints pinned against the clamp
permanently — a bang-bang controller with essentially no proportional control. It is
self-reinforcing: once actions saturate, extra exploration noise no longer changes behavior, so the
entropy bonus inflates $\log\sigma$ for free, driving further saturation.

**This silently explains several earlier mysteries.** Responding to fine reward shaping requires
fine control, and there wasn't any left:

- v6's air-time penalty moved nothing across 52M steps.
- Speed tracking was compressed to roughly a single gait: commands spanning $0.5$–$2.0$ m/s
  produced only $0.93$–$1.59$ m/s actual.
- Entropy plateaued near 40 nats ($\sigma \approx 3.6$) and would not come down.

**It is not repairable after the fact.** Rescaling a saturated policy's output layer back into
range destroys the gait, because the behavior is *built on* saturated commands:

| output scale | ep_len | distance |
|---|---|---|
| 1.00 (baseline) | 949 | 9.99 m |
| 0.50 | 156 | 0.91 m |
| 0.25 | 22 | 0.03 m |

So this has to be *prevented during training*, not fixed later.

**The barrier.** $w_{\text{sat}}$ is deliberately tiny, making this a barrier rather than a cost —
invisible while the policy behaves, severe if it runs away:

| commanded $\lvert a\rvert$ | penalty/step | vs. max velocity reward |
|---|---|---|
| 1.0 | $-0.0010$ | 0.1% |
| 5.0 | $-0.0250$ | 2.5% |
| 24.0 | $-0.5760$ | 57.6% |
| 50.0 | $-2.5000$ | 250% |

Physics is unaffected (raw $1.0$ and raw $50.0$ both clamp to $1.0$ → bit-identical states).

⚠️ **Requires a fresh run.** The existing checkpoint pays $-0.71$/step under this term — about 75%
of its total $+0.94$ — so resuming from it would just crush a policy that cannot climb back down.

---

## Current constants

All in [`reward_terms.py`](reward_terms.py), shared by both backends so they cannot drift apart.

| symbol | constant | value |
|---|---|---|
| $\sigma$ | `FORWARD_VEL_TRACKING_SCALE` | 1.0 |
| $v^{*}$ | `COMMAND_MIN/MAX_SPEED` | 0.5 – 2.0 m/s |
| — | `ALIVE_BONUS` | 0.1 |
| $w_{\text{orient}}$ | `ORIENTATION_WEIGHT` | 0.5 |
| $w_{\omega}$ | `ANGULAR_VEL_WEIGHT` | 0.05 |
| $w_{\text{bal}}$ | `BALANCE_WEIGHT` | 0.5 |
| $w_y, w_z$ | `LATERAL/VERTICAL_VEL_WEIGHT` | 0.1 |
| $w_{\text{gait}}$ | `GAIT_PHASE_WEIGHT` | 0.3 |
| $T_{\text{gait}}$ | `GAIT_CYCLE_TIME` | 0.8 s |
| $t^{*}$ | `TARGET_AIR_TIME` | $T_{\text{gait}}/2$ = 0.4 s |
| $t_{\max}$ | `MAX_AIR_TIME` | 0.45 s |
| $w_{\text{excess}}$ | `AIR_TIME_EXCESS_WEIGHT` | 2.0 |
| $w_{\text{slip}}$ | `FOOT_SLIP_WEIGHT` | 0.1 |
| $c_{\text{energy}}$ | `ENERGY_COST_WEIGHT` | 0.005 |
| $w_{\ddot q}$ | `JOINT_ACCEL_WEIGHT` | $1\times10^{-8}$ |
| $w_{\dot q}$ | `JOINT_VEL_WEIGHT` | $5\times10^{-3}$ |
| $w_{\Delta a}$ | `ACTION_RATE_WEIGHT` | 0.25 |
| $P_{\text{fall}}$ | `FALL_PENALTY` | 20.0 |

The four effort/smoothness terms are **means** over joints, not sums. As sums they scaled with joint
count, so expanding the action space from 5 (legs only) to 15 (whole body) silently tripled their
weight — taxing *having* more limbs rather than moving them wastefully. Weights are $5\times$ their
original values (the joint count they were calibrated at) so a mean lands where that calibration
intended.

## Measured breakdown (850M policy)

Per-step average over 5,452 real steps — the shape you want, objective dominant and penalties
secondary:

| term | value | | term | value |
|---|---|---|---|---|
| `velocity_tracking` | $+0.8780$ | | `action_rate` | $-0.0346$ |
| `gait_phase` | $+0.3416$ | | `balance` | $-0.0285$ |
| `alive` | $+0.1000$ | | `feet_air_time` | $-0.0214$ |
| `angular_velocity` | $-0.1000$ | | `vertical_velocity` | $-0.0163$ |
| `foot_slip` | $-0.0587$ | | `orientation` | $-0.0096$ |
| `joint_velocity` | $-0.0562$ | | `energy` | $-0.0043$ |
| `lateral_velocity` | $-0.0430$ | | `joint_acceleration` | $-0.0038$ |
| | | | `fall` | $-0.0037$ |
| | | | **TOTAL** | $\mathbf{+0.9397}$ |

---

## Lessons

1. **Measure the term, don't reason about it.** Both the slip bug ($-0.42$/step) and the dead
   velocity gradient ($0.0001$) were invisible in `ep_rew_mean` and obvious the instant terms were
   plotted separately. This is why per-term TensorBoard logging exists.
2. **A penalty must be large enough to change the trade.** v6 is correct and useless.
3. **Check what a term actually measures, not what it's named.** v3 read the root joint while the
   body bent at the waist; v4's slip read COM velocity, not foot velocity.
4. **Derive related constants from each other.** $t^{*}$ and $T_{\text{gait}}$ silently disagreed
   while set independently.
5. **If every term is a penalty, dying is optimal.** Always check the sign of a typical step.
6. **Penalize what the policy commands, not what survives clamping.** Anything outside the
   clamp is invisible to the reward and therefore free to grow without limit.
