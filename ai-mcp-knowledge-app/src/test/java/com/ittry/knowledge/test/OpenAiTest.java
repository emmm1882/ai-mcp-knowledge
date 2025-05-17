package com.ittry.knowledge.test;

import com.alibaba.fastjson.JSON;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.model.Media;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.Resource;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.util.MimeType;
import org.springframework.util.MimeTypeUtils;
import reactor.core.publisher.Flux;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Collectors;

@Slf4j
@RunWith(SpringRunner.class)
@SpringBootTest
public class OpenAiTest {

    @Value("classpath:data/dog.png")
    private Resource imageResource;

    @Autowired
    private OpenAiChatModel openAiChatModel;

    @jakarta.annotation.Resource(name = "openAiSimpleVectorStore")
    private SimpleVectorStore simpleVectorStore;

    @jakarta.annotation.Resource(name = "openAiPgVectorStore")
    private PgVectorStore pgVectorStore;

    @jakarta.annotation.Resource
    private TokenTextSplitter tokenTextSplitter;

    @jakarta.annotation.Resource
    private OpenAiApi openAiApi;

    @Test
    public void test_call() {
        ChatResponse response = openAiChatModel.call(new Prompt(
                "1+1",
                OpenAiChatOptions.builder()
                        .model("gpt-4o")
                        .build()));

        log.info("测试结果(call):{}", JSON.toJSONString(response));
    }

    @Test
    public void test_call_images() {
        // 构建请求信息
        UserMessage userMessage = new UserMessage("请描述这张图片的主要内容，并说明图中物品的可能用途。",
                new Media(MimeType.valueOf(MimeTypeUtils.IMAGE_PNG_VALUE),
                        imageResource));

        ChatResponse response = openAiChatModel.call(new Prompt(
                userMessage,
                OpenAiChatOptions.builder()
                        .model("gpt-4o")
                        .build()));

        log.info("测试结果(images):{}", JSON.toJSONString(response));
    }

    @Test
    public void test_stream() throws InterruptedException {
        CountDownLatch countDownLatch = new CountDownLatch(1);

        Flux<ChatResponse> stream = openAiChatModel.stream(new Prompt(
                "1+1",
                OpenAiChatOptions.builder()
                        .model("gpt-4o")
                        .build()));

        stream.subscribe(
                chatResponse -> {
                    AssistantMessage output = chatResponse.getResult().getOutput();
                    log.info("测试结果(stream): {}", JSON.toJSONString(output));
                },
                Throwable::printStackTrace,
                () -> {
                    countDownLatch.countDown();
                    log.info("测试结果(stream): done!");
                }
        );

        countDownLatch.await();
    }

    @Test
    public void upload() {
        TikaDocumentReader reader = new TikaDocumentReader("./data/file.txt");

        List<Document> documents = reader.get();
        List<Document> documentSplitterList = tokenTextSplitter.apply(documents);

        documents.forEach(doc -> doc.getMetadata().put("knowledge", "知识库名称v2"));
        documentSplitterList.forEach(doc -> doc.getMetadata().put("knowledge", "知识库名称v2"));

//        pgVectorStore.accept(documentSplitterList);

        log.info("上传完成");
    }

    @Test
    public void chat() {
        String message = "人工智能是什么";

        String SYSTEM_PROMPT = """
                Use the information from the DOCUMENTS section to provide accurate answers but act as if you knew this information innately.
                If unsure, simply state that you don't know.
                Another thing you need to note is that your reply must be in Chinese!
                DOCUMENTS:
                    {documents}
                """;

        SearchRequest request = SearchRequest.builder()
                .query(message)
                .topK(5)
                .filterExpression("knowledge == '知识库名称v2'")
                .build();

        List<Document> documents = pgVectorStore.similaritySearch(request);

        String documentsCollectors = null == documents ? "" : documents.stream().map(Document::getText).collect(Collectors.joining());

        Message ragMessage = new SystemPromptTemplate(SYSTEM_PROMPT).createMessage(Map.of("documents", documentsCollectors));

        ArrayList<Message> messages = new ArrayList<>();
        messages.add(new UserMessage(message));
        messages.add(ragMessage);

        ChatResponse chatResponse = openAiChatModel.call(new Prompt(
                messages,
                OpenAiChatOptions.builder()
                        .model("gpt-4o")
                        .build()));

        log.info("测试结果:{}", JSON.toJSONString(chatResponse));
    }

    @Test
    public void test_() {
    }

}
