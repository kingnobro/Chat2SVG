export interface WebSocketMessage {
  stage?: number;
  status?: string;
  output?: string;
  target?: string;
  timestamp?: number;
  svg_file?: string;
  message?: string;
}
export {};