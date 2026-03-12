"""
UrbanFlow AI - AI Traffic Assistant (Explainable AI)
====================================================
Page 5: Natural Language Chat Interface allowing users to query the AI logic engine.
"""
import streamlit as st
import time

def show():
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">AI Traffic Assistant</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Ask questions to understand the AI Traffic Brain's real-time decisions and routing strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the UrbanFlow AI Traffic Brain. How can I help you understand the current city traffic state?"}
        ]

    st.markdown('<div class="saas-card" style="padding-bottom:12px; min-height:500px; display:flex; flex-direction:column;">', unsafe_allow_html=True)

    # Context scrolling container
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Pre-defined smart queries
    cols = st.columns(4)
    suggested_queries = [
        "Why did the signal change at INT-02?",
        "Why was the green corridor activated?",
        "Which intersection has the highest congestion?",
        "What is the fastest route to the hospital?"
    ]
    
    st.markdown("<div style='margin-top:16px; margin-bottom:8px; font-size:12px; font-weight:600; color:#64748B; text-transform:uppercase;'>Suggested Queries</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    buttons = [c1, c2, c3, c4]
    
    for i, q in enumerate(suggested_queries):
        if buttons[i].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.shortcut_query = q
            st.rerun()

    prompt = st.chat_input("Ask the AI Traffic Brain a question...")
    
    # Handle shortcut triggers over native input
    if "shortcut_query" in st.session_state and st.session_state.shortcut_query:
        prompt = st.session_state.shortcut_query
        st.session_state.shortcut_query = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Simulate Processing latency
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🧠 *Analyzing traffic telemetry...*")
                time.sleep(1.0) # Artificial thinking gap
                
                # Intelligent Keyword Mapping Strategy (Explainable AI Engine)
                p_lower = prompt.lower()
                response = ""
                
                if "int-02" in p_lower or "signal" in p_lower:
                    response = "Heavy traffic was detected in the north lane of **INT-02**. To counteract the volume, I dynamically extended the signal duration to **45 seconds** to clear the queue and reduce upstream congestion."
                elif "green corridor" in p_lower or "ambulance" in p_lower or "emergency" in p_lower:
                    response = "An ambulance was detected at **INT-03**. The system immediately activated a **Green Priority Corridor** and synchronized all downstream signals along the optimal route to the City Hospital to guarantee zero-stop transit."
                elif "congestion" in p_lower or "highest" in p_lower:
                    response = "Currently, **INT-04** is experiencing the highest localized congestion (89% Capacity Load). I am actively routing approaching northbound vehicles to INT-02 to distribute the network pressure."
                elif "fastest route" in p_lower or "hospital" in p_lower:
                    response = "Based on live density mapping, the current fastest route to the **City Hospital** originates from the East quadrant passing through **INT-06 -> INT-05 -> Hospital**. This avoids the heavy accident congestion accumulating near the northern limits."
                else:
                    response = "I am currently monitoring 6 active intersections and 4 live camera feeds. The global traffic network is operating within nominal parameters. Traffic density is flowing optimally."
                
                message_placeholder.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
