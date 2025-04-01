import { Box, Typography, Button } from '@mui/material';

interface ProgressProps {
  currentStage: number;
  failedStage?: number | null;
  onRetry?: (stage: number) => void;
}

const stages = ['Template Generation', 'Detail Enhancement', 'SVG Shape Optimization'];

export default function ProgressStepper({ currentStage, failedStage, onRetry }: ProgressProps) {
  return (
    <Box sx={{ mt: 2 }}>
      {stages.map((label, index) => {
        const stageNumber = index + 1;
        const isFailed = stageNumber === failedStage;
        const isCompleted = (stageNumber < currentStage) && !isFailed;
        const isCurrent = currentStage === stageNumber && !isFailed;

        return (
          <Box
            key={index}
            sx={{
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              mb: 1,
              p: 1,
              bgcolor: getStatusColor(currentStage, stageNumber, failedStage),
              borderRadius: 50,
              minWidth: {xs: '100%', md: 1200}
            }}>

            <Box sx={{
              width: 24,
              height: 24,
              borderRadius: '50%',
              bgcolor: getStageColor(isCompleted, isFailed),
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mr: 3
            }}>
              {isCompleted && '✓'}
              {isFailed && '×'}
            </Box>


            <Typography variant="body2" sx={{ flexGrow: 1 }}>
              {label}
              {isCurrent && '(running)'}
              {isFailed && '(error)'}
            </Typography>

            {isFailed && (
              <Button
                variant="outlined"
                size="small"
                sx={{
                  position: 'absolute',
                  right: 8,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  fontSize: '0.75rem',
                  padding: '2px 8px',
                  lineHeight: 1.2,
                  minWidth: 'auto'
                }}
                onClick={() => onRetry?.(stageNumber)}
              >
                retry
              </Button>
            )}
          </Box>
        );
      })}
    </Box>
  );
}


const getStatusColor = (currentStage: number, stage: number, failedStage?: number | null) => {
  if (stage === failedStage) return '#ffebee';
  return currentStage >= stage ? '#e8f5e9' : 'transparent';
};

const getStageColor = (isCompleted: boolean, isFailed: boolean) => {
  if (isFailed) return '#ff1744';
  return isCompleted ? '#4caf50' : '#ddd';
};