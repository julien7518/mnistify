"use client";

import {
  Card,
  CardDescription,
  CardTitle,
  CardHeader,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SelectGroup,
  SelectLabel,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { ChartView, chartDataProps } from "@/components/chartview";
import { useRef, useState, useEffect } from "react";
import cnn from "@/public/cnn/cnn.js";
import mlp from "@/public/mlp/mlp.js";
import { Status, StatusIndicator, StatusLabel } from "@/components/ui/status";
import { Spinner } from "@/components/ui/spinner";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<"draw" | "erase">("draw");
  const [selectedModel, setSelectedModel] = useState<"cnn" | "mlp" | "">(
    "",
  );
  const [isDrawing, setIsDrawing] = useState(false);
  const [chartData, setChartData] = useState<chartDataProps>(
    Array.from({ length: 10 }, (_, i) => ({ number: i, proba: 0 })),
  );
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [loadTimeMs, setLoadTimeMs] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [device, setDevice] = useState<GPUDevice | null>(null);
  const [modelInference, setModelInference] = useState<
    ((input: Float32Array) => Promise<Float32Array[]>) | null
  >(null);

  // Initialisation de WebGPU
  useEffect(() => {
    const initWebGPU = async () => {
      if (!navigator.gpu) {
        console.error("WebGPU not supported");
        return;
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.error("No GPU adapter found");
        return;
      }
      const gpuDevice = await adapter.requestDevice();
      setDevice(gpuDevice);
    };
    initWebGPU();
  }, []);

  // Initialisation du canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = 280;
    canvas.height = 280;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  // Chargement du modèle sélectionné
  useEffect(() => {
    const loadModel = async () => {
      if (!selectedModel || !device) return;

      setIsLoadingModel(true);
      setLoadTimeMs(null);
      const startTime = performance.now();

      try {
        const modelPath =
          selectedModel === "cnn"
            ? "/cnn/cnn.webgpu.safetensors"
            : "/mlp/mlp.webgpu.safetensors";

        const response = await fetch(modelPath);
        if (!response.ok) {
          throw new Error(response.statusText);
        }

        const arrayBuffer = await response.arrayBuffer();
        const safetensorData = new Uint8Array(arrayBuffer);

        const modelLoader = selectedModel === "cnn" ? cnn : mlp;
        const inference = await modelLoader.setupNet(device, safetensorData);

        setModelInference(() => inference);
        const loadTime = Math.round(performance.now() - startTime);
        setLoadTimeMs(loadTime);
      } catch (error) {
        console.error("Error loading model:", error);
        if (error instanceof Error) {
          setErrorMessage(error.message);
        }
      } finally {
        setIsLoadingModel(false);
      }
    };

    loadModel();
  }, [selectedModel, device]);

  // Prétraitement de l'image pour le modèle
  const preprocessCanvas = (): Float32Array => {
    const canvas = canvasRef.current;
    const previewCanvas = previewCanvasRef.current;
    if (!canvas) return new Float32Array(784);

    const ctx = canvas.getContext("2d");
    if (!ctx) return new Float32Array(784);

    // Créer un canvas temporaire 28x28
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext("2d");
    if (!tempCtx) return new Float32Array(784);

    // Redimensionner l'image à 28x28
    tempCtx.fillStyle = "white";
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);

    // Extraire les pixels (image originale)
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;

    const input = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      const pixelValue =
        (pixels[i * 4] * 0.299) + (pixels[i * 4 + 1] * 0.587) + (pixels[i * 4 + 2] * 0.114);
      input[i] = (255 - pixelValue) / 255.0;
    }

    // Construire une image inversée pour l'affichage (pour montrer ce que le modèle voit)
    const invertedImageData = tempCtx.createImageData(28, 28);
    const invPixels = invertedImageData.data;
    for (let i = 0; i < 784; i++) {
      const gray =
        (pixels[i * 4] * 0.299) + (pixels[i * 4 + 1] * 0.587) + (pixels[i * 4 + 2] * 0.114);
      const inv = 255 - Math.round(gray);
      invPixels[i * 4] = inv;
      invPixels[i * 4 + 1] = inv;
      invPixels[i * 4 + 2] = inv;
      invPixels[i * 4 + 3] = 255;
    }
    // Remplacer le contenu du tempCanvas par l'image inversée
    tempCtx.putImageData(invertedImageData, 0, 0);

    // Afficher l'aperçu 28x28 (agrandi à 280x280) en pixelated afin de montrer l'entrée du modèle
    if (previewCanvas) {
      const previewCtx = previewCanvas.getContext("2d");
      if (previewCtx) {
        previewCtx.clearRect(0, 0, 280, 280);
        previewCtx.imageSmoothingEnabled = false;
        previewCtx.drawImage(tempCanvas, 0, 0, 28, 28, 0, 0, 280, 280);
      }
    }

    return input;
  };

  // Prédiction après le dessin
  const runPrediction = async () => {
    if (!modelInference) return;

    try {
      const input = preprocessCanvas();
      const output = await modelInference(input);

      // Softmax pour obtenir les probabilités
      const logits = Array.from(output[0]);
      const maxLogit = Math.max(...logits);
      const exps = logits.map((x) => Math.exp(x - maxLogit));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probabilities = exps.map((x) => x / sumExps);

      // Mettre à jour le graphique
      const newChartData = probabilities.map((proba, i) => ({
        number: i,
        proba: Math.round(proba * 100),
      }));
      setChartData(newChartData);
    } catch (error) {
      console.error("Error during prediction:", error);
    }
  };

  // Obtenir les coordonnées relatives au canvas
  const getCoordinates = (
    e:
      | React.MouseEvent<HTMLCanvasElement>
      | React.TouchEvent<HTMLCanvasElement>,
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();

    if ("touches" in e) {
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top,
      };
    } else {
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    }
  };

  // Début du dessin
  const startDrawing = (
    e:
      | React.MouseEvent<HTMLCanvasElement>
      | React.TouchEvent<HTMLCanvasElement>,
  ) => {
    e.preventDefault();
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) return;

    const { x, y } = getCoordinates(e);

    if (mode === "draw") {
      ctx.strokeStyle = "black";
      ctx.lineWidth = 20;
    } else {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 40;
    }

    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  // Dessin en cours
  const draw = (
    e:
      | React.MouseEvent<HTMLCanvasElement>
      | React.TouchEvent<HTMLCanvasElement>,
  ) => {
    e.preventDefault();
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) return;

    const { x, y } = getCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  // Fin du dessin avec prédiction
  const stopDrawing = () => {
    if (isDrawing && modelInference) {
      runPrediction();
    }
    setIsDrawing(false);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) return;
    ctx.closePath();
  };

  // Effacer tout le canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) return;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Effacer le canvas de prévisualisation
    const previewCanvas = previewCanvasRef.current;
    if (previewCanvas) {
      const previewCtx = previewCanvas.getContext("2d");
      if (previewCtx) {
        previewCtx.fillStyle = "white";
        previewCtx.fillRect(0, 0, 280, 280);
      }
    }

    // Réinitialiser les prédictions
    setChartData(
      Array.from({ length: 10 }, (_, i) => ({ number: i, proba: 0 })),
    );
  };

  return (
    <main className="min-h-screen flex items-start justify-center">
      <div className="w-full max-w-4xl space-y-6 px-4 py-8">
        <Card className="w-full">
          <CardHeader>
            <CardTitle className="flex justify-around text-2xl">
              MNISTify
            </CardTitle>
            <CardDescription className="flex justify-around">
              First, select an AI model. Then sketch a number on the canvas and
              let it predict which digit you drew.
            </CardDescription>
            <CardDescription className="flex justify-around">
              <p>You can find the GitHub repo <a className="underline" href="https://github.com/julien7518/mnistify" target="_blank">here.</a></p>
            </CardDescription>
          </CardHeader>
        </Card>

        {/* Sélecteur de modèle */}
        <Select value={selectedModel} onValueChange={(value: string) => setSelectedModel(value as "cnn" | "mlp" | "")}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Models</SelectLabel>
              <SelectItem value="cnn">Convolutional Neural Network</SelectItem>
              <SelectItem value="mlp">Multi-Layer Perceptron</SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>

        {/* Canvas de dessin et prévisualisation */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Draw Here</CardTitle>
              <div className="flex justify-between w-full pt-2">
                <ButtonGroup>
                  <Button
                    variant={mode === "draw" ? "secondary" : "outline"}
                    onClick={() => setMode("draw")}
                    disabled={!modelInference}
                  >
                    Draw
                  </Button>
                  <Button
                    variant={mode === "erase" ? "secondary" : "outline"}
                    onClick={() => setMode("erase")}
                    disabled={!modelInference}
                  >
                    Erase
                  </Button>
                </ButtonGroup>
                <ButtonGroup>
                  <Button
                    variant="destructive"
                    onClick={clearCanvas}
                    disabled={!modelInference}
                  >
                    Clear
                  </Button>
                </ButtonGroup>
              </div>
            </CardHeader>
            <div className="flex justify-center p-4">
              <canvas
                ref={canvasRef}
                className="border border-gray-300 rounded-lg bg-white cursor-crosshair touch-none"
                style={{ opacity: modelInference ? 1 : 0.5 }}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
            </div>
            <CardDescription className="text-center pb-4">
              {selectedModel !== "" ? (
                isLoadingModel ? (
                  <div className="inline-flex items-center justify-center gap-2">
                    <Spinner className="h-4 w-4" />
                    <span>Loading model...</span>
                  </div>
                ) : (
                  <Status
                    status={loadTimeMs !== null ? "online" : "offline"}
                    className="inline-flex items-center bg-transparent"
                  >
                    <StatusIndicator />
                    <StatusLabel time={loadTimeMs} error={errorMessage} />
                  </Status>
                )
              ) : (
                <p>Select a model to start</p>
              )}
            </CardDescription>
          </Card>

          {/* Canvas de prévisualisation 28x28 */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Model Input (28×28)</CardTitle>
              <CardDescription className="py-[12px]">This is what the AI model sees.</CardDescription>
            </CardHeader>
            <div className="flex justify-center p-4">
              <canvas
                ref={previewCanvasRef}
                width={280}
                height={280}
                className="border border-gray-300 rounded-lg bg-white"
                style={{ imageRendering: "pixelated" }}
              />
            </div>
          </Card>
        </div>

        {/* Graphique des probabilités */}
        <Card className="px-16">
          <CardHeader>
            <CardDescription className="flex justify-around">
              {chartData.some((d) => d.proba > 0) ? (
                <ChartView data={chartData} />
              ) : (
                <p>Start drawing to see predictions</p>
              )}
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    </main>
  );
}
