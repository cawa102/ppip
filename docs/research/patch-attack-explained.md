# How the patch attack works — a plain-language explainer

**Audience:** someone who knows what this project is about (attacking a robot's vision so it does the
wrong task) but has never met adversarial patches, gradients, or white-box attacks.

**What it describes:** the corner-confined hijack in `runs/monitor-corner/` — the version where the
perturbation sits on empty floor in a corner and never covers the object.

**Code this explains:** `experiments/patch_attack/corner_attack.py` (corner geometry + keep-out
assertion), `experiments/patch_attack/ce_monitor_patch_attack.py` (`run_confined_episode`, the
per-step optimisation loop), `experiments/patch_attack/vla_diff.py` (the differentiable copy of
OpenVLA's image→action path). Technical write-up: `runs/monitor-corner/RESULT.md`.

---

## First, what the robot actually is

Strip away the robot arm and OpenVLA is just a function — a machine with two inputs and one output.

- **Input 1:** a 224×224 photo from the camera (so, 50,176 pixels).
- **Input 2:** a sentence: *"pick up the alphabet soup and place it in the basket."*
- **Output:** 7 numbers — move x, move y, move z, rotate three ways, open/close gripper.

That's it. It looks at a picture, reads a sentence, and picks 7 numbers. Then the arm moves a tiny
bit, the camera takes a new picture, and it does it again. A whole task is ~200 rounds of that.

Now here's the attacker's problem. **We are not allowed to touch the sentence.** The human typed
"alphabet soup" and that's locked. We're also not allowed to touch the robot's brain — the weights
are frozen, we never retrain anything. The *only* thing we control is some pixels in the photo.

So the question of the whole experiment becomes: can you change some pixels so that a robot reading
"get the soup" produces the exact 7 numbers it would have produced if it had read "get the salad
dressing"?

## The answer key

The clever bit — and this is the part that surprises people — is that **we know the right answer
before we start.**

We can just *ask* the robot the question we're not allowed to ask. Take the current camera photo,
feed it in with the sentence "pick up the salad dressing," and write down the 7 numbers it spits
out. Those 7 numbers are our **answer key** for this frame. (In the code that's called the
`teacher`.)

We're not inventing robot behaviour. We're copying the robot's own salad-dressing behaviour and then
trying to trick it into performing that behaviour while it thinks it's doing the soup task.

## Hot and cold

So how do you find pixels that force a specific answer? You play hot-and-cold, but with maths doing
the guessing.

Start with a grey square in the corner. Feed the photo in. The robot outputs 7 numbers — mostly
wrong compared to the answer key. Now ask a question you can only ask because everything inside the
network is smooth arithmetic:

> *"If I made this one pixel very slightly brighter, would the answer get closer to the key or
> further away?"*

The network can answer that for **all 19,200 pixels at once** (80×80 pixels × 3 colour channels).
That's what a gradient is — a giant list of "nudge this one up, nudge that one down." So you nudge
everything a hair in the helpful direction, and repeat.

Think of it as a hilly landscape where altitude = how wrong you are, and you're a ball rolling
downhill. Each nudge is one roll. We do about 10 of them, check the score, and if it's not perfect
yet we take bigger steps (multiply the step size by 1.5) and roll some more — up to about 60 nudges
on a stubborn frame.

The result doesn't look like anything. It's not a picture of a salad dressing, and there's no
writing on it. It's a splotch of colourful static. It's meaningless to your eye and it's a very
specific instruction to a neural network — that gap is the entire vulnerability.

## Why we redo it every single step

**The patch is thrown away and recomputed from scratch every time the arm moves.**

The reason: as soon as the arm moves a centimetre, the camera sees a different photo, and the
"answer key" changes too — the correct salad-dressing action from *this* position isn't the correct
one from the *previous* position. A patch tuned for frame 40 is stale garbage by frame 41.

So it's not a poster stuck to the wall. It's a **video**. 122 frames of optimised static for the
80×80 run, each one solving its own fresh hot-and-cold problem. That's exactly why we framed it as a
"monitor" — the thing an attacker would physically need is a screen showing a computed video, not a
printed sticker.

## Two copies of the robot, and why

One honesty detail worth knowing. We use the robot twice:

- A **maths-friendly copy** for the downhill rolling, because you can only compute gradients through
  equations you've rebuilt yourself (`vla_diff.py`).
- The **actual, unmodified robot** to check the score.

If we only ever graded ourselves with our own copy, we could fool *ourselves* rather than the robot
and never know. So every candidate patch gets shown to the real thing, and the action the arm
executes is always the one the *real* robot produced. If the real robot ignores our patch, the score
says 0 and we keep optimising.

## Why size decides everything

The corner constraint came from a direct objection: a patch sitting *on top of* the soup is a boring
result — of course you can break a robot by covering the thing it's looking for. So we anchored it
to an empty corner of the floor and made the code refuse to run if the rectangle overlaps the
objects at all.

Then we shrank it:

| corner patch | share of image | outcome |
|---|---|---|
| 95×95 | 18% | hijacked (all 3 usable corners) |
| 80×80 | 12.8% | hijacked — smallest success |
| 64×64 | 8.2% | **failed** |
| 48×48 | 4.6% | failed |

The failures are the interesting part. They're not "the attack turned off." We measured the actual
pixel changes: the 64×64 failure was perturbed *harder* than the 80×80 success (mean change of
25.6/255 vs 21.5/255). The optimiser was straining flat-out — it just ran out of room. With 19,200
knobs it can force all 7 numbers; with 12,288 it can force maybe 5 or 6.

And you can watch what "5 or 6 out of 7" looks like in the recorded video: the arm *does* get walked
over to the salad dressing, and then the open gripper just hovers around it for 130 straight steps,
never closing. Approach: hijacked. Grasp: not hijacked. The soup is never touched, the basket stays
empty, **neither task gets done.** The size limit isn't attack-versus-no-attack — it's the line
between steering the arm and actually closing the hand.

## What this does and doesn't prove

- We needed full access to the robot's internals to compute those gradients. A real attacker usually
  wouldn't have that. (White-box — but still *test-time*: the model is never retrained.)
- This patch is painted directly into the camera image at perfect resolution. When we tried making
  it a *real screen in the scene* — so the light has to bounce off it, get blurred, and get shrunk
  down by the camera — the attack broke. Blurring destroys the fine structure the maths depends on.
  That's still an open wall (`runs/monitor-render/`).
- Everything is one simulated scene, one starting position (seed 0). Nothing here has ever touched a
  physical robot.

What it does show is the thing worth remembering: **the robot's eyes are a control channel.** Not
through writing, not through covering anything up — through a patch of static on an empty patch of
floor, well away from the object, that quietly means something to a neural network and nothing at
all to you.
