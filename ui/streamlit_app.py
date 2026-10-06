"""Streamlit UI for the privacy & safety gateway.

Run with: streamlit run ui/streamlit_app.py

Prompts are kept only in the browser session (st.session_state); they
are never written to files or logs.
"""

import html
import os
from typing import Any

import httpx
import streamlit as st

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")
_TIMEOUT = 30.0

_VERDICT_STYLE = {"allow": st.success, "warn": st.warning, "block": st.error}
_OVERRIDE_OPTIONS = ["keep policy action", "allow", "mask", "redact", "tokenize"]


def _get(path: str) -> Any | None:
    try:
        with httpx.Client(base_url=GATEWAY_URL, timeout=_TIMEOUT) as client:
            resp = client.get(path)
    except httpx.HTTPError:
        st.error("Could not reach the gateway. Is the server running?")
        return None
    if resp.status_code == 404:
        st.error("Not found (404). Check the gateway URL.")
        return None
    if resp.status_code >= 500:
        st.error(f"Gateway error (HTTP {resp.status_code}). Please try again.")
        return None
    if resp.status_code != 200:
        st.error(f"Unexpected response (HTTP {resp.status_code}).")
        return None
    return resp.json()


def _post(path: str, payload: dict[str, Any]) -> Any | None:
    try:
        with httpx.Client(base_url=GATEWAY_URL, timeout=_TIMEOUT) as client:
            resp = client.post(path, json=payload)
    except httpx.HTTPError:
        st.error("Could not reach the gateway. Is the server running?")
        return None
    if resp.status_code == 404:
        st.error("Unknown profile or route (404).")
        return None
    if resp.status_code == 422:
        st.error("Invalid request (422). Check the prompt and overrides.")
        return None
    if resp.status_code == 502:
        st.error("The AI backend failed (502). Please try again later.")
        return None
    if resp.status_code != 200:
        st.error(f"Unexpected response (HTTP {resp.status_code}).")
        return None
    return resp.json()


def _fetch_profiles() -> list[str]:
    data = _get("/v1/profiles")
    if data is None:
        return []
    return list(data.get("profiles", []))


def _highlight(prompt: str, findings: list[dict[str, Any]]) -> str:
    """Render prompt with findings highlighted; everything is escaped."""
    parts: list[str] = []
    cursor = 0
    for f in sorted(findings, key=lambda x: x["start"]):
        if f["start"] < cursor:
            continue
        parts.append(html.escape(prompt[cursor : f["start"]]))
        label = html.escape(f"{f['type']} ({f['confidence']:.2f})")
        parts.append(
            f'<mark title="{label}" style="background:#ffd966">'
            f"{html.escape(prompt[f['start'] : f['end']])}</mark>"
        )
        cursor = f["end"]
    parts.append(html.escape(prompt[cursor:]))
    return "".join(parts)


def _override_controls(findings: list[dict[str, Any]]) -> dict[str, str]:
    """One selectbox per finding type; 'block' is never offered."""
    overrides: dict[str, str] = {}
    seen: set[str] = set()
    for f in findings:
        type_name = f["type"]
        if type_name in seen:
            continue
        seen.add(type_name)
        if f["action"] == "block":
            st.caption(f"{type_name}: blocked by policy")
            continue
        choice = st.selectbox(
            f"Override for {type_name}",
            _OVERRIDE_OPTIONS,
            key=f"override_{type_name}",
        )
        if choice != "keep policy action":
            overrides[type_name] = choice
    return overrides


def _render_analysis(prompt: str, analysis: dict[str, Any]) -> dict[str, str]:
    injection = analysis["injection"]
    style = _VERDICT_STYLE.get(injection["verdict"], st.info)
    style(f"Injection verdict: **{injection['verdict']}** (score {injection['score']})")

    findings = analysis["findings"]
    if findings:
        st.markdown(_highlight(prompt, findings), unsafe_allow_html=True)
        rows = [
            {
                "type": f["type"],
                "position": f"{f['start']}:{f['end']}",
                "confidence": f["confidence"],
                "policy action": f["action"],
            }
            for f in findings
        ]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No findings.")
    return _override_controls(findings)


def _review_tab() -> None:
    prompt = st.text_area("Prompt", height=150)
    profiles = _fetch_profiles()
    profile = st.selectbox("Profile", profiles) if profiles else None

    if st.button("Analyze"):
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            analysis = _post("/v1/analyze", {"prompt": prompt, "profile": profile})
            if analysis is not None:
                st.session_state["analysis"] = analysis
                st.session_state["prompt"] = prompt
                st.session_state["profile"] = profile

    analysis = st.session_state.get("analysis")
    if analysis is not None:
        stored_prompt = st.session_state.get("prompt", prompt)
        overrides = _render_analysis(stored_prompt, analysis)

        if st.button("Send to AI"):
            result = _post(
                "/v1/chat",
                {
                    "prompt": stored_prompt,
                    "profile": st.session_state.get("profile"),
                    "overrides": overrides,
                },
            )
            if result is not None:
                st.session_state["chat_result"] = result

    chat_result = st.session_state.get("chat_result")
    if chat_result is not None:
        st.subheader("Result")
        st.write(f"Decision: **{chat_result.get('decision')}**")
        if chat_result.get("reason"):
            st.write(f"Reason: {chat_result['reason']}")
        if chat_result.get("blocked_types"):
            st.write("Blocked types: " + ", ".join(chat_result["blocked_types"]))
        if chat_result.get("applied"):
            st.write("Applied actions:")
            st.dataframe(chat_result["applied"], use_container_width=True)
        if chat_result.get("reply"):
            st.markdown("**Reply:**")
            st.write(chat_result["reply"])


def _dashboard_tab() -> None:
    if st.button("Refresh"):
        st.session_state.pop("summary", None)
    summary = st.session_state.get("summary")
    if summary is None:
        summary = _get("/v1/audit/summary")
        if summary is not None:
            st.session_state["summary"] = summary
    if summary is None:
        return

    st.metric("Total events", summary.get("total_events", 0))

    counts_per_type = summary.get("counts_per_type", {})
    if counts_per_type:
        st.subheader("Findings per type")
        st.bar_chart(counts_per_type)

    decisions = summary.get("decisions", {})
    if decisions:
        st.subheader("Decisions")
        st.bar_chart(decisions)

    blocked_per_day = summary.get("blocked_per_day", {})
    if blocked_per_day:
        st.subheader("Blocked per day")
        st.bar_chart(blocked_per_day)


def main() -> None:
    st.title("Privacy & Safety Gateway")
    review_tab, dashboard_tab = st.tabs(["Review and send", "Audit dashboard"])
    with review_tab:
        _review_tab()
    with dashboard_tab:
        _dashboard_tab()


main()
