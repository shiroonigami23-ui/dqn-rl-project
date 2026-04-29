using UnityEngine;
using System;

public class CartPoleArenaManager : MonoBehaviour
{
    public CartPoleAgent dqnAgent;
    public CartPoleAgent randomAgent;
    public int seed = 42;
    public bool autoReset = true;
    public float resetDelaySec = 1.0f;

    public int episodeIndex = 0;
    public event Action<int,float,float,int,int> OnEpisodeFinished;

    private float resetTimer = 0f;
    private System.Random rng;
    private bool signaled = false;

    void Start()
    {
        rng = new System.Random(seed);
        ResetEpisodeBoth();
    }

    void FixedUpdate()
    {
        if (dqnAgent == null || randomAgent == null) return;

        if (!dqnAgent.Done) dqnAgent.Tick();
        if (!randomAgent.Done) randomAgent.Tick();

        if (dqnAgent.Done && randomAgent.Done)
        {
            if (!signaled)
            {
                signaled = true;
                OnEpisodeFinished?.Invoke(
                    episodeIndex,
                    dqnAgent.EpisodeReward,
                    randomAgent.EpisodeReward,
                    dqnAgent.StepCount,
                    randomAgent.StepCount
                );
            }

            if (autoReset)
            {
                resetTimer += Time.fixedDeltaTime;
                if (resetTimer >= resetDelaySec)
                {
                    resetTimer = 0f;
                    episodeIndex++;
                    ResetEpisodeBoth();
                }
            }
        }
    }

    public void ResetEpisodeBoth()
    {
        float x = Rand(-0.05f, 0.05f);
        float xd = Rand(-0.05f, 0.05f);
        float t = Rand(-0.05f, 0.05f);
        float td = Rand(-0.05f, 0.05f);

        dqnAgent.ResetWithState(x, xd, t, td);
        randomAgent.ResetWithState(x, xd, t, td);
        signaled = false;
    }

    float Rand(float a, float b)
    {
        return (float)(a + rng.NextDouble() * (b - a));
    }
}
