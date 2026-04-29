using UnityEngine;

public class CartPoleAgent : MonoBehaviour
{
    [Header("Scene References")]
    public Transform cart;
    public Transform pole;
    public DQNInference dqn;

    [Header("Identity")]
    public string agentName = "DQN";
    public bool useRandomPolicy = false;

    [Header("Physics")]
    public float gravity = 9.8f;
    public float cartMass = 1.0f;
    public float poleMass = 0.1f;
    public float poleHalfLength = 0.5f;
    public float forceMag = 10.0f;
    public float tau = 0.02f;

    [Header("Limits")]
    public float xThreshold = 2.4f;
    public float thetaThresholdRad = 12f * Mathf.Deg2Rad;
    public int maxSteps = 500;

    [Header("Runtime")]
    public int seed = 42;

    private float x, xDot, theta, thetaDot;
    private int stepCount;
    private float episodeReward;
    private bool done;
    private System.Random rng;
    private float[] lastQ = new float[2];

    public int StepCount => stepCount;
    public float EpisodeReward => episodeReward;
    public bool Done => done;
    public float X => x;
    public float Theta => theta;
    public float ThetaDeg => theta * Mathf.Rad2Deg;
    public float[] LastQ => lastQ;

    void Start()
    {
        rng = new System.Random(seed);
    }

    public void ResetWithState(float ix, float ixd, float it, float itd)
    {
        x = ix; xDot = ixd; theta = it; thetaDot = itd;
        stepCount = 0;
        episodeReward = 0f;
        done = false;
        RenderState();
    }

    public void Tick()
    {
        if (done) return;

        int action;
        if (useRandomPolicy)
        {
            action = rng.Next(0, 2);
            lastQ[0] = float.NaN;
            lastQ[1] = float.NaN;
        }
        else
        {
            float[] state = new float[] { x, xDot, theta, thetaDot };
            lastQ = dqn.Predict(state);
            action = lastQ[0] > lastQ[1] ? 0 : 1;
        }

        StepPhysics(action);
        RenderState();

        bool terminated = Mathf.Abs(x) > xThreshold || Mathf.Abs(theta) > thetaThresholdRad;
        bool truncated = stepCount >= maxSteps;

        if (!terminated) episodeReward += 1f;

        if (terminated || truncated)
        {
            done = true;
            Debug.Log($"{agentName} done | reward={episodeReward} | steps={stepCount}");
        }
    }

    void StepPhysics(int action)
    {
        float force = action == 1 ? forceMag : -forceMag;
        float totalMass = cartMass + poleMass;
        float poleMassLen = poleMass * poleHalfLength;

        float cosTheta = Mathf.Cos(theta);
        float sinTheta = Mathf.Sin(theta);

        float temp = (force + poleMassLen * thetaDot * thetaDot * sinTheta) / totalMass;
        float thetaAcc = (gravity * sinTheta - cosTheta * temp) /
                         (poleHalfLength * (4f / 3f - poleMass * cosTheta * cosTheta / totalMass));
        float xAcc = temp - poleMassLen * thetaAcc * cosTheta / totalMass;

        x += tau * xDot;
        xDot += tau * xAcc;
        theta += tau * thetaDot;
        thetaDot += tau * thetaAcc;

        stepCount += 1;
    }

    void RenderState()
    {
        if (cart != null)
        {
            Vector3 p = cart.position;
            cart.position = new Vector3(x, p.y, p.z);
        }
        if (pole != null && cart != null)
        {
            pole.position = cart.position + new Vector3(0f, 0.5f, 0f);
            pole.rotation = Quaternion.Euler(0f, 0f, -theta * Mathf.Rad2Deg);
        }
    }
}
