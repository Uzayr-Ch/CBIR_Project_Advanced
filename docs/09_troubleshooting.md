# 09 — Troubleshooting

## Common Issues

| Problem | Cause | Solution |
|:--------|:------|:---------|
| `ModuleNotFoundError` | Package not installed in venv | `pip install <module>` inside activated venv |
| `features.pkl not found` | Features haven't been extracted | Run `python extractor.py` |
| `combined_features.pkl not found` | Combined index not built | Run `python batch_evaluate.py` first |
| Slow feature extraction | CPU-based processing | Normal: ~1-2 images/sec on CPU |
| Image won't load | Corrupt file or bad path | Run `python validate_data.py` |
| `No module named 'sklearn'` | scikit-learn missing | `pip install scikit-learn` |
| `No module named 'matplotlib'` | matplotlib missing | `pip install matplotlib` |
| Streamlit not found | Not installed in venv | `pip install streamlit` |
| Wrong Python running | Global Python instead of venv | Make sure venv is activated first |

---

## Environment Issues

### PowerShell activation blocked

```powershell
# Fix execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Rebuild virtual environment

```powershell
python -m venv cbir_env --clear
.\cbir_env\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit numpy pillow scikit-learn tqdm matplotlib
```

### Verify you're in the right environment

```powershell
# Should show path inside cbir_env
python -c "import sys; print(sys.executable)"
```

---

## Streamlit Issues

### App crashes on load

- Check terminal for error messages
- Most common: missing package → install it
- If using `use_container_width`: update to `width="stretch"`

### Cache stale data

- After changing feature files, Streamlit may use cached old data
- Fix: Press `Ctrl+Shift+R` in browser (hard refresh)
- Or restart Streamlit: `Ctrl+C` then `streamlit run app.py` again

### Port already in use

```powershell
streamlit run app.py --server.port 8502
```

---

## Memory Issues

| Dataset | Approx Memory Needed |
|:--------|:------|
| Corel-1K | ~50 MB |
| Corel-5K | ~200 MB |
| Corel-10K | ~400 MB |
| Combined (25K) | ~800 MB |

If you run out of memory, evaluate datasets one at a time instead of loading the combined index.

---

*Previous: [08 - Usage Guide](08_usage_guide.md) | Next: [10 - References](10_references.md)*
