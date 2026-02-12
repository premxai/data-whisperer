#!/bin/bash
# Wait for Ollama to be ready and pull the model if needed

if [ "$LLM_PROVIDER" = "ollama" ] || [ -z "$LLM_PROVIDER" ]; then
    echo "Waiting for Ollama to start..."
    MAX_WAIT=60
    WAITED=0
    while ! curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "Ollama not available after ${MAX_WAIT}s, starting without LLM (fallback mode)"
            break
        fi
    done

    if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "Ollama is up. Checking for model ${OLLAMA_MODEL}..."
        if ! curl -s "$OLLAMA_HOST/api/tags" | grep -q "$OLLAMA_MODEL"; then
            echo "Pulling ${OLLAMA_MODEL}... (this may take a few minutes on first run)"
            curl -s "$OLLAMA_HOST/api/pull" -d "{\"name\": \"$OLLAMA_MODEL\"}"
            echo "Model pull complete."
        else
            echo "Model ${OLLAMA_MODEL} already available."
        fi
    fi
else
    echo "Using cloud LLM provider: $LLM_PROVIDER"
fi

exec "$@"
