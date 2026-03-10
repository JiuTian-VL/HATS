# AndroidWorld Evaluation Integration

## 1. Copy Evaluation Files into AndroidWorld

Copy the following files into `android_world/agents/` in your AndroidWorld repository:

- `evaluate/Android/device.py`
- `evaluate/Android/HATS_AndroidWorld.py`

## 2. Modify AndroidWorld `run.py`

Add the following import at the top of the file:

```python
from android_world.agents import HATS_AndroidWorld
```

Then add the following block to the `_get_agent` function in `run.py`:

```python
elif _AGENT_NAME.value == "HATS":
        # model: str = "qwen2_vl"
        # model: str = "qwen2_5_vl"
        # model: str = "internvl2_4b"
        model: str = "internvl2_8b"
        openai_base_url: str = "http://127.0.0.1:8000/v1"
        documents: dict[str, str] = {}
        documents_fp = "/path/to/all_documents.pkl.zst"
        from android_world.agents.device import load_object_from_disk
        documents = load_object_from_disk(documents_fp)
        agent = HATS_AndroidWorld.HATSAgent(
            env=env,
            model=model,
            openai_base_url=openai_base_url,
            documents=documents,
        )
```

(Optional) Replace `documents_fp` with the actual local path on your machine. After running `python fusion_all_documents.py`, you will get `all_documents.pkl.zst`.

## 3. Run the Evaluation

For the remaining steps, follow the official AndroidWorld documentation.
