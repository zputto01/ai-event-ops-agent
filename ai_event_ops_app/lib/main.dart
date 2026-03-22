import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

const String apiBaseUrl = 'https://ai-event-ops-agent-xeps5klwnq-nw.a.run.app';

void main() {
  runApp(const AiEventOpsApp());
}

class AiEventOpsApp extends StatelessWidget {
  const AiEventOpsApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Event Ops Agent',
      theme: ThemeData(useMaterial3: true),
      home: const ChatScreen(),
    );
  }
}

class ChatMessage {
  final String role; // "user" | "assistant"
  final String text;
  final List<Citation> citations;
  final String confidence; // "answer" | "escalate"
  final String? reason;

  ChatMessage({
    required this.role,
    required this.text,
    this.citations = const [],
    this.confidence = 'answer',
    this.reason,
  });
}

class Citation {
  final String doc;
  final int? page;
  final String? section;
  final String chunkId;

  Citation({
    required this.doc,
    required this.chunkId,
    this.page,
    this.section,
  });

  factory Citation.fromJson(Map<String, dynamic> json) {
    return Citation(
      doc: json['doc'] as String,
      page: json['page'] as int?,
      section: json['section'] as String?,
      chunkId: json['chunk_id'] as String,
    );
  }

  String display() {
    final parts = <String>[doc];
    if (page != null) parts.add('p.$page');
    if (section != null && section!.trim().isNotEmpty) parts.add('[$section]');
    return parts.join(' ');
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<ChatMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  bool _isSending = false;

  // Simple session id for Firestore escalation grouping
  final String _sessionId = 'flutter-${DateTime.now().millisecondsSinceEpoch}';

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _isSending) return;

    setState(() {
      _isSending = true;
      _messages.add(ChatMessage(role: 'user', text: text));
      _controller.clear();
    });
    _scrollToBottom();

    try {
      final resp = await http.post(
        Uri.parse('$apiBaseUrl/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message': text,
          'session_id': _sessionId,
        }),
      );

      if (resp.statusCode != 200) {
        setState(() {
          _messages.add(ChatMessage(
            role: 'assistant',
            text: 'Server error (${resp.statusCode}).',
            confidence: 'escalate',
            reason: resp.body.isNotEmpty ? resp.body : 'Non-200 response',
          ));
        });
        _scrollToBottom();
        return;
      }

      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      final confidence = (data['confidence'] as String?) ?? 'escalate';
      final answer = (data['answer'] as String?) ?? '';
      final reason = data['reason'] as String?;
      final citationsJson = (data['citations'] as List<dynamic>? ?? []);
      final citations = citationsJson
          .map((c) => Citation.fromJson(c as Map<String, dynamic>))
          .toList();

      setState(() {
  if (confidence == 'answer') {
    _messages.add(ChatMessage(
      role: 'assistant',
      text: answer,
      citations: citations,
      confidence: confidence,
      reason: null,
    ));
  } else {
    final friendlyReason = _friendlyReason(reason);

    _messages.add(ChatMessage(
      role: 'assistant',
      text: 'I couldn’t find a reliable answer to that in the event documents.',
      citations: const [],
      confidence: 'escalate',
      reason: friendlyReason,
    ));
  }
});
      _scrollToBottom();
    } catch (e) {
      setState(() {
        _messages.add(ChatMessage(
  role: 'assistant',
  text: 'Something went wrong while contacting the assistant.',
  confidence: 'escalate',
  reason: 'Please try again in a moment.',
));
      });
      _scrollToBottom();
    } finally {
      setState(() => _isSending = false);
    }
  }

  void _scrollToBottom() {
    // Give the list a beat to render
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent + 200,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  Widget _bubble(ChatMessage msg) {
    final isUser = msg.role == 'user';
    final bg = isUser
        ? Theme.of(context).colorScheme.primaryContainer
        : Theme.of(context).colorScheme.surfaceContainerHighest;

    final border = (!isUser && msg.confidence == 'escalate')
        ? Border.all(color: Theme.of(context).colorScheme.error, width: 1)
        : null;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: bg,
            borderRadius: BorderRadius.circular(14),
            border: border,
          ),
          child: Column(
            crossAxisAlignment:
                isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
            children: [
  Text(msg.text),
  if (!isUser && msg.confidence == 'answer' && msg.citations.isNotEmpty)
    Padding(
      padding: const EdgeInsets.only(top: 10),
      child: _citationsView(msg.citations),
    ),
  if (!isUser && msg.confidence == 'escalate' && msg.reason != null)
    Padding(
      padding: const EdgeInsets.only(top: 10),
      child: Text(
        'Helpful note: ${msg.reason}',
        style: TextStyle(
          color: Theme.of(context).colorScheme.error,
          fontSize: 12,
        ),
      ),
    ),
],
          ),
        ),
      ),
    );
  }

  Widget _citationsView(List<Citation> citations) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Citations',
          style: TextStyle(fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 6),
        ...citations.map((c) => Text(
              '• ${c.display()}',
              style: const TextStyle(fontSize: 12),
            )),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Event Ops Agent'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(22),
          child: Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(
              'Session: $_sessionId',
              style: Theme.of(context).textTheme.labelSmall,
            ),
          ),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: _messages.length,
              itemBuilder: (_, i) => _bubble(_messages[i]),
            ),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.all(10),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    onSubmitted: (_) => _send(),
                    decoration: const InputDecoration(
                      hintText: 'Ask an exhibitor/speaker question...',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: _isSending ? null : _send,
                  child: _isSending
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Send'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
String _friendlyReason(String? rawReason) {
  if (rawReason == null || rawReason.isEmpty) {
    return 'Try asking about build-up, breakdown, deliveries, stand power, health and safety, or speaker logistics.';
  }

  if (rawReason.contains('Low retrieval confidence')) {
    return 'That question doesn’t seem to relate to the exhibitor manual. Try asking about event operations, logistics, or rules.';
  }

  if (rawReason.contains('insufficient context')) {
    return 'I couldn’t find enough detail in the documents to answer that. Try being more specific.';
  }

  return rawReason; // fallback
}
