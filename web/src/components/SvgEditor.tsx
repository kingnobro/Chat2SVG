import {useState} from 'react';
import {Box, Button, TextField, Alert} from '@mui/material';

interface SvgEditorProps {
    content: string;
}

const sanitizeSVG = (svg: string) => {
    return svg
        .replace(/<script.*?>.*?<\/script>/gi, '')
        .replace(/on\w+=".*?"/g, '');
};

export default function SvgEditor({content}: SvgEditorProps) {
    const [previewMode, setPreviewMode] = useState(true);

    const handleDownload = () => {
        const sanitized = sanitizeSVG(content);
        const blob = new Blob([sanitized], {type: 'image/svg+xml'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `design-${Date.now()}.svg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <Box sx={{
            border: '1px solid #ddd',
            borderRadius: 1,
            flexShrink: 0,
            width: '100%',
            maxWidth: 600,
            mb: 2
        }}>
            <Box sx={{
                p: 2,
                bgcolor: '#f5f5f5',
                display: 'flex',
                gap: 1,
                alignItems: 'center'
            }}>
                <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setPreviewMode(true)}
                    disabled={previewMode}
                >
                    Preview
                </Button>
                <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setPreviewMode(false)}
                    disabled={!previewMode}
                >
                    Code
                </Button>
                <Button
                    variant="contained"
                    size="small"
                    onClick={handleDownload}
                    sx={{marginLeft: 'auto'}}
                >
                    Download
                </Button>
            </Box>

            <Box sx={{p: 2}}>
                {previewMode ? (
                    <Box sx={{
                        border: '1px solid #eee',
                        padding: 2,
                        '& svg': {
                            maxWidth: '100%',
                            height: 'auto'
                        }
                    }}>
                        <div dangerouslySetInnerHTML={{__html: sanitizeSVG(content)}}/>
                    </Box>
                ) : (
                    <TextField
                        fullWidth
                        multiline
                        minRows={10}
                        maxRows={20}
                        value={content}
                        InputProps={{
                            readOnly: true,
                        }}
                        variant="outlined"
                    />
                )}
            </Box>
        </Box>
    );
}