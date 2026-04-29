using UnityEngine;

public class CameraShake : MonoBehaviour
{
    public CartPoleAgent sourceAgent;
    public float amplitude = 0.07f;
    public float frequency = 14f;

    private Vector3 baseLocalPos;

    void Start()
    {
        baseLocalPos = transform.localPosition;
    }

    void LateUpdate()
    {
        if (sourceAgent == null)
        {
            transform.localPosition = baseLocalPos;
            return;
        }

        float intensity = Mathf.Clamp01(Mathf.Abs(sourceAgent.ThetaDeg) / 12f) * amplitude;
        float t = Time.time * frequency;
        Vector3 jitter = new Vector3(Mathf.Sin(t) * intensity, Mathf.Cos(t * 1.31f) * intensity, 0f);
        transform.localPosition = baseLocalPos + jitter;
    }
}
