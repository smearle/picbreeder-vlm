/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.tools;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

import javax.xml.parsers.*;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.*;

import client.utilities.ArgumentParser;

/**
 * The below is taken from the GeneticArt's About file:
 * 
 * Inspired by the GeneticArt tool from mattias fagerlund (which is based on delphiNEAT, 
 * http://www.cambrianlabs.com/mattias/GeneticArt) i just implement a c# version 
 * of it (based on sharpNeat, http://sharpneat.sourceforge.net/). Just for fun. 
 *
 * This program converts their XML files into a format compatible with WebNEAT.
 *    
 * @author Adam Campbell
 */
public class GeneticArtConverter {
	public static void main(String[] args){
		if(args.length == 0) {
			System.out.println("Usage: java client.tools.GeneticArtConverter [options]");
			System.out.println("Options:");
			System.out.println("    -o outputFile");
			System.out.println("    -i inputFile");
			return;
		}

		ArgumentParser argParser = new ArgumentParser(args);
		
		String inputFile = argParser.findArgument("-i");
		String outputFile = argParser.findArgument("-o");
		String seriesId, actFunction;
		
		try {
			Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(new File(inputFile));
			Document output = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
			
			Element parentElement, currentElement, genomeElement, delphiGenome = doc.getDocumentElement();
			NodeList list;
			
			seriesId = "geneticArtImport" + delphiGenome.getAttribute("id");
			actFunction = delphiGenome.getAttribute("activation-fn-id");
			
			parentElement = output.createElement("series");
			parentElement.setAttribute("branch", seriesId);
			output.appendChild(parentElement);
			
			currentElement = output.createElement("generation");
			currentElement.setAttribute("size", "1");
			parentElement.appendChild(currentElement);
			
			parentElement = currentElement;
			genomeElement = output.createElement("genome");
			genomeElement.setAttribute("age", "1");
			parentElement.appendChild(genomeElement);
			
			currentElement = output.createElement("identifier");
			currentElement.setAttribute("branch", seriesId);
			currentElement.setAttribute("id", "1");
			genomeElement.appendChild(currentElement);
			
			currentElement = output.createElement("parents");
			currentElement.setAttribute("count", "0");
			genomeElement.appendChild(currentElement);
			
			if(actFunction.equals("Gausian")) actFunction = "gaussian(x)";
			else actFunction = "sigmoid(x)";
				
			list = ((Element)delphiGenome.getElementsByTagName("neurons").item(0)).getElementsByTagName("neuron");
			currentElement = output.createElement("nodes");
			currentElement.setAttribute("count", ""+list.getLength());
			genomeElement.appendChild(currentElement);
			parentElement = currentElement;
			
			String[] inputs = new String[]{"x", "y", "d"};
			int inputIndex = 0;
			for(int i = 0; i < list.getLength(); i++){
				String id = ((Element)list.item(i)).getAttribute("id");
				String type = ((Element)list.item(i)).getAttribute("type");
				
				Element nodeElement = output.createElement("node");
				parentElement.appendChild(nodeElement);
				
				if(type.equals("in"))
					nodeElement.setAttribute("label", inputs[inputIndex++]);
				else if(type.equals("out"))
					nodeElement.setAttribute("label", "ink");
				else if(type.equals("bias"))
					nodeElement.setAttribute("label", "bias");
				
				if(type.equals("in") || type.equals("bias"))
					nodeElement.setAttribute("type", "in");
				else if(type.equals("out"))
					nodeElement.setAttribute("type", "out");
				else if(type.equals("hid"))
					nodeElement.setAttribute("type", "hidden");
				else
					throw new RuntimeException("Unknown node type: " + type);
				
				currentElement = output.createElement("marking");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+id);
				nodeElement.appendChild(currentElement);
				
				currentElement = output.createElement("activation");
				if(type.equals("bias") || type.equals("in"))
					currentElement.setTextContent("identity(x)");
				else
					currentElement.setTextContent(actFunction);
				nodeElement.appendChild(currentElement);
				
			}
			
			list = ((Element)delphiGenome.getElementsByTagName("connections").item(0)).getElementsByTagName("connection");
			currentElement = output.createElement("links");
			currentElement.setAttribute("count", ""+list.getLength());
			genomeElement.appendChild(currentElement);
			parentElement = currentElement;
			
			for(int i = 0; i < list.getLength(); i++){
				String id = ((Element)list.item(i)).getAttribute("innov-id");
				String srcId = ((Element)list.item(i)).getAttribute("src-id");
				String tgtId = ((Element)list.item(i)).getAttribute("tgt-id");
				String weight = ((Element)list.item(i)).getAttribute("weight");
				
				Element connElement = output.createElement("link");
				parentElement.appendChild(connElement);
				
				currentElement = output.createElement("marking");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+id);
				connElement.appendChild(currentElement);
				
				currentElement = output.createElement("source");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+srcId);
				connElement.appendChild(currentElement);
				
				currentElement = output.createElement("target");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+tgtId);
				connElement.appendChild(currentElement);
				
				currentElement = output.createElement("weight");
				currentElement.setTextContent(""+weight);
				connElement.appendChild(currentElement);
			}
			
			TransformerFactory transFactory = TransformerFactory.newInstance();
			transFactory.setAttribute("indent-number", 4);
			Transformer trans = transFactory.newTransformer();
			trans.setOutputProperty(OutputKeys.INDENT, "yes");
			trans.transform(new DOMSource(output), new StreamResult(new OutputStreamWriter(new FileOutputStream(outputFile))));
			
		}catch(Exception e) {
			throw new RuntimeException("XML Errors");
		}
		
	}
}
