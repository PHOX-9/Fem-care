import express from "express";
import authRoutes from "./routes/auth";
import cors from "cors";
import { corsOptions } from "./config/CorsConfig";
import onboardingRoutes from "./modules/onboarding/onboarding.routes";

const app = express();

app.use(express.json());
app.use(cors(corsOptions));
app.use("/auth", authRoutes);
app.use("/onboarding", onboardingRoutes);

export default app;
