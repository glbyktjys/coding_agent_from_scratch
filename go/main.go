package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
	"github.com/openai/openai-go/v3"
)

const (
	model               = openai.ChatModelGPT5
	maxCompletionTokens = 2000

	youColor       = "\033[94m"
	assistantColor = "\033[93m"
	resetColor     = "\033[0m"
)

const systemPromptTemplate = `
You are a coding assistant whose goal it is to help us solve coding tasks.
You have access to a series of tools you can execute. Here are the tools you can execute:

%s

When you want to use a tool, reply with exactly one line in the format: 'tool: TOOL_NAME({JSON_ARGS})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.
`

// tool is something the model can invoke by replying with `tool: name({...})`.
type tool struct {
	name        string
	description string
	args        string // JSON shape of the arguments, shown to the model
	run         func(args json.RawMessage) (any, error)
}

// tools is a slice rather than a map so the system prompt lists them in a stable order.
var tools = []tool{
	{
		name:        "read_file",
		description: "Gets the full content of a file provided by the user.",
		args:        `{"filename": string}`,
		run:         readFileTool,
	},
	{
		name:        "list_file",
		description: "Lists all files in a directory provided by the user.",
		args:        `{"path": string}`,
		run:         listFilesTool,
	},
	{
		name: "edit_file",
		description: "Replaces first occurrence of old_str with new_str in file. " +
			"If old_str is empty, creates/overwrites file with new_str.",
		args: `{"path": string, "old_str": string, "new_str": string}`,
		run:  editFileTool,
	},
}

func findTool(name string) (tool, bool) {
	for _, t := range tools {
		if t.name == name {
			return t, true
		}
	}
	return tool{}, false
}

// resolveAbsPath turns "file.go" or "~/file.go" into an absolute path.
func resolveAbsPath(path string) (string, error) {
	if path == "~" || strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		path = filepath.Join(home, path[1:])
	}
	return filepath.Abs(path)
}

func readFileTool(raw json.RawMessage) (any, error) {
	var args struct {
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, err
	}
	path, err := resolveAbsPath(args.Filename)
	if err != nil {
		return nil, err
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return map[string]any{"filename": args.Filename, "content": string(content)}, nil
}

func listFilesTool(raw json.RawMessage) (any, error) {
	var args struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, err
	}
	path, err := resolveAbsPath(args.Path)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	files := make([]map[string]string, 0, len(entries))
	for _, entry := range entries {
		kind := "file"
		if entry.IsDir() {
			kind = "dir"
		}
		files = append(files, map[string]string{"filename": entry.Name(), "type": kind})
	}
	return map[string]any{"path": path, "files": files}, nil
}

func editFileTool(raw json.RawMessage) (any, error) {
	var args struct {
		Path   string `json:"path"`
		OldStr string `json:"old_str"`
		NewStr string `json:"new_str"`
	}
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, err
	}
	path, err := resolveAbsPath(args.Path)
	if err != nil {
		return nil, err
	}
	result := func(action string) map[string]any {
		return map[string]any{"path": path, "action": action}
	}

	if args.OldStr == "" {
		if err := os.WriteFile(path, []byte(args.NewStr), 0o644); err != nil {
			return nil, err
		}
		return result("created_file"), nil
	}
	original, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	before, after, found := strings.Cut(string(original), args.OldStr)
	if !found {
		return result("old_str not found"), nil
	}
	edited := before + args.NewStr + after
	if err := os.WriteFile(path, []byte(edited), 0o644); err != nil {
		return nil, err
	}
	return result("edited"), nil
}

func buildSystemPrompt() string {
	var toolList strings.Builder
	for _, t := range tools {
		fmt.Fprintf(&toolList, "Name: %s\nDescription: %s\nArguments: %s\n%s\n",
			t.name, t.description, t.args, strings.Repeat("=", 15))
	}
	return fmt.Sprintf(systemPromptTemplate, toolList.String())
}

type toolCall struct {
	name string
	args json.RawMessage
}

// extractToolCalls returns the tool calls requested in `tool: name({...})` lines.
// The parser expects single-line, compact JSON in parentheses.
func extractToolCalls(text string) []toolCall {
	var calls []toolCall
	for line := range strings.Lines(text) {
		after, ok := strings.CutPrefix(strings.TrimSpace(line), "tool:")
		if !ok {
			continue
		}
		name, rest, ok := strings.Cut(after, "(")
		if !ok {
			continue
		}
		argsJSON, ok := strings.CutSuffix(rest, ")")
		if !ok || !json.Valid([]byte(argsJSON)) {
			continue
		}
		calls = append(calls, toolCall{name: strings.TrimSpace(name), args: json.RawMessage(argsJSON)})
	}
	return calls
}

// runToolCall executes a call and returns its result as JSON. Failures are
// reported back to the model instead of crashing the agent.
func runToolCall(call toolCall) string {
	var result any
	err := fmt.Errorf("unknown tool: %s", call.name)
	if t, ok := findTool(call.name); ok {
		result, err = t.run(call.args)
	}
	if err != nil {
		result = map[string]string{"error": err.Error()}
	}
	// Results only hold strings, slices and maps, so marshaling can't fail.
	encoded, _ := json.Marshal(result)
	return string(encoded)
}

func callLLM(ctx context.Context, client *openai.Client, messages []openai.ChatCompletionMessageParamUnion) (string, error) {
	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:               model,
		Messages:            messages,
		MaxCompletionTokens: openai.Int(maxCompletionTokens),
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("response has no choices")
	}
	return resp.Choices[0].Message.Content, nil
}

func runAgentLoop(ctx context.Context, client *openai.Client) error {
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(buildSystemPrompt()),
	}
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("%sYou%s: ", youColor, resetColor)
		if !scanner.Scan() { // EOF (Ctrl+D) or read error
			fmt.Println()
			return scanner.Err()
		}
		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			continue
		}
		messages = append(messages, openai.UserMessage(userInput))

		// Keep calling the model until it replies without requesting a tool.
		for {
			reply, err := callLLM(ctx, client, messages)
			if err != nil {
				fmt.Fprintln(os.Stderr, "LLM call failed:", err)
				break
			}
			fmt.Printf("%sAssistant%s: %s\n", assistantColor, resetColor, reply)
			messages = append(messages, openai.AssistantMessage(reply))

			calls := extractToolCalls(reply)
			if len(calls) == 0 {
				break
			}
			for _, call := range calls {
				fmt.Printf("-> %s(%s)\n", call.name, call.args)
				messages = append(messages, openai.UserMessage("tool_result("+runToolCall(call)+")"))
			}
		}
	}
}

func main() {
	// Optional: OPENAI_API_KEY may already be exported in the shell.
	_ = godotenv.Load("../.env")
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY not set (export it or add it to ../.env)")
	}
	client := openai.NewClient() // reads OPENAI_API_KEY from the environment
	if err := runAgentLoop(context.Background(), &client); err != nil {
		log.Fatal(err)
	}
}
