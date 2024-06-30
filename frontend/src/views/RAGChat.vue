<template>
  <a-row :span="24">
    <a-col :span="12">
      <a-card :bordered="true" :style="{ width: '100%' }">
        <ChatUI/>
      </a-card>
    </a-col>
    <a-col :span="12">
      <a-card :bordered="true" :style="{ width: '100%' }">
        <ChatDetails/>
      </a-card>
    </a-col>
  </a-row>
</template>
<script setup>
import ChatDetails from "@/components/RAGChat/ChatDetails.vue";
import ChatUI from "@/components/RAGChat/ChatUI.vue";
</script>