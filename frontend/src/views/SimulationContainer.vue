<template>
  <a-row :span="24">
    <a-col :span="12">
      <a-card :bordered="true" :style="{ width: '100%' }">
        <ChatUI/>
      </a-card>
    </a-col>
    <a-col :span="12">
      <a-card :bordered="true" :style="{ width: '100%' }">
        <SimulationUI/>
      </a-card>
    </a-col>
  </a-row>
</template>
<script setup>
import ChatUI from "@/components/SimulationContainer/ChatUI.vue";
import SimulationUI from "@/components/SimulationContainer/SimulationUI.vue";
</script>